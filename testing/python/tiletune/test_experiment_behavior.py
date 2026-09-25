"""Work accounting against the authoritative experiment kernels."""

import pytest

from experiments.common.kernels import make_case
from experiments.common.spec import Workload, default_workloads
from tilelang.tiletune.compute import operation_work
from tilelang.tiletune.src.collector import _Collector


def example(name, stages=0):
    workload = (
        Workload("kda_chunk_tails", "kda_chunk_o", dict(batch=1, heads=4, sequence=768, dim=96, value_dim=80, chunk_size=48))
        if name == "kda_chunk_tails"
        else next(w for w in default_workloads() if w.name == name)
    )
    if workload.op in ("gemm", "gemm_fp8"):
        config = dict(block_M=64, block_N=64, block_K=32, num_stages=stages, enable_rasteration=False)
        config["thread_num" if workload.op == "gemm" else "threads"] = 128
    elif workload.op == "attention":
        config = dict(block_M=64, block_N=64, num_stages=stages, threads=128)
    else:
        config = dict(block_DK=32, block_DV=32, num_stages=stages, threads=128)
    case = make_case(workload)
    return workload, case, case.build(**config)


@pytest.mark.parametrize("name", [w.name for w in default_workloads() if w.op not in ("gemm_fp8", "grouped_gemm")])
def test_example_shared_output_path_is_charged(name):
    _, _, func = example(name)
    operations = _Collector(func).operations
    store = operations[-1]
    stage = operations[-2]
    shared = stage.writes[0].buffer
    elements = 1
    for extent in shared.shape:
        elements *= int(extent)
    assert shared.scope().startswith("shared")
    assert store.reads[0].buffer.same_as(shared)
    # Fragment -> shared followed by shared -> global: one internal write
    # and one shared read, in addition to the separately modeled global store.
    assert operation_work(stage)["shared_bytes"] == elements * 2
    assert operation_work(store)["shared_bytes"] == elements * 2


@pytest.mark.parametrize("name", ["kda_chunk_regular", "kda_chunk_tails"])
def test_kda_gate_reads_both_dtypes_and_writes_rounded_query(name):
    workload, _, func = example(name)
    gate = next(op for op in _Collector(func).operations if op.kind == "elementwise")
    work = operation_work(gate)
    elements = workload.parameters["chunk_size"] * 32
    assert work["exp_ops"] == elements
    # Q is FP16, G is FP32, gated Q is FP16; this is per key tile.
    assert work["shared_bytes"] == elements * (2 + 4 + 2)


def test_async_input_copies_retain_the_producer_transfer_accounting():
    _, _, func = example("kda_chunk_regular", stages=2)
    producers = [op for op in _Collector(func).operations if any(r.buffer.scope() == "global" for r in op.reads)]
    assert producers
    assert all(operation_work(op)["shared_bytes"] == 0 for op in producers)


@pytest.mark.parametrize("name", [w.name for w in default_workloads() if w.op != "grouped_gemm"] + ["kda_chunk_tails"])
@pytest.mark.parametrize("stages", [0, 2])
def test_every_final_workload_models_its_actual_loop_and_memory(name, stages):
    from math import ceil

    from tilelang.tiletune import analyze_prim_func
    from test_ampere import profile
    from test_portability import AMPERE, LIMITS

    workload, case, func = example(name, stages)
    p = workload.parameters
    rates, target = profile(), AMPERE
    if workload.op == "gemm_fp8":
        from tilelang.tiletune.profiling.device_profile import _signature

        target = dict(kind="cuda", arch="sm_90a")
        rates.update(
            profile_target="sm_90a",
            gemm_signature=_signature(workload.dtype, "float32"),
            convert_float32_to_float8_e4m3fn_per_cycle=64,
        )
    before = func.script()
    result = analyze_prim_func(
        func,
        dict(ranking_metric="pipeline_time", performance_model=rates),
        target=target,
        device_limits=LIMITS,
        pass_configs=case.pass_configs,
    )
    assert func.script() == before
    assert result["tile_cost"]["score"] is not None, result["tile_cost"].get("unknown")
    pipe = result["modules"]["pipeline_overlap"]
    memory = result["modules"]["memory_traffic"]
    matrix = [phase for phase in pipe["phases"] if phase["work"]["gemm_flops"]]
    if workload.op in ("gemm", "gemm_fp8"):
        width = 1 if workload.op == "gemm_fp8" else 2
        assert len(matrix) == 1 and matrix[0]["work"]["gemm_flops"] == 2 * 64 * 64 * 32
        assert pipe["iterations"]["max"] == ceil(p["k"] / 32)
        expected_read = (64 + 64) * p["k"] * width
        expected_write = 64 * 64 * width
        if workload.op == "gemm_fp8":
            assert sum(phase["work"].get("convert_float32_to_float8_e4m3fn", 0) for phase in pipe["phases"]) == 64 * 64
    elif workload.op == "attention":
        assert len(matrix) == 2
        assert all(phase["inside_loop"] for phase in matrix)
        assert [phase["work"]["gemm_flops"] for phase in matrix] == [2 * 64 * 64 * p["dim"]] * 2
        assert pipe["iterations"]["max"] == ceil(p["sequence"] / 64)
        assert pipe["iterations"]["min"] == (1 if p["causal"] else ceil(p["sequence"] / 64))
        expected_read = (64 + 2 * p["sequence"]) * p["dim"] * 2
        expected_write = 64 * p["dim"] * 2
    else:
        s = p["chunk_size"]
        assert len(matrix) == 2
        assert [phase["inside_loop"] for phase in matrix] == [True, False]
        assert [phase["work"]["gemm_flops"] for phase in matrix] == [2 * s * 32 * 32, 2 * s * s * 32]
        expected_read = s * p["dim"] * 6 + p["dim"] * 32 * 2 + s * 32 * 2 + s * s * 2
        expected_write = s * 32 * 2
    assert memory["input_bytes_per_block"] == expected_read
    assert memory["output_bytes_per_block"] == expected_write

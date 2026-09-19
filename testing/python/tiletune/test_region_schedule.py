"""Independent work-count and region transition regressions."""

import pytest
import tilelang.language as T
from regression_kernels import softmax_program
from test_ampere import analyze


@pytest.mark.parametrize("rows,columns", [(8, 256), (7, 259), (1, 17)])
def test_two_pass_softmax_counts_each_masked_access(rows, columns):
    result = analyze(softmax_program(rows, columns, "float16", 4, 128, 128, 1, 4))
    pipeline = result["modules"]["pipeline_overlap"]
    assert not pipeline["unknown"], pipeline
    assert pipeline["timing"] is not None
    totals = pipeline["region_work"]
    distribution = pipeline["cta_work"]
    reads = sum(totals[g["iterations"]]["read_bytes"] * g["count"] for g in distribution["groups"]) * distribution["repetitions"]
    writes = sum(totals[g["iterations"]]["write_bytes"] * g["count"] for g in distribution["groups"]) * distribution["repetitions"]
    # Enumerate real input/output coordinates independently of region geometry.
    elements = sum(1 for _ in range(rows) for _ in range(columns))
    assert reads == 2 * elements * 2
    assert writes == elements * 2
    assert len([n for n in pipeline["region_schedule"]["variants"][0] if "runs" in n]) == 2


def test_causal_prefix_then_guarded_tail():
    @T.prim_func
    def kernel(X: T.Tensor((7, 13), "float32"), Y: T.Tensor((7,), "float32")):
        with T.Kernel(4, threads=128) as bx:
            tile = T.alloc_fragment((2, 4), "float32")
            total = T.alloc_fragment((2,), "float32")
            T.clear(total)
            for k in T.serial(T.ceildiv(T.min((bx + 1) * 3, 13), 4)):
                T.copy(X[bx * 2, k * 4], tile)
                for i in T.Parallel(2):
                    total[i] += tile[i, 0]
            for i in T.Parallel(2):
                if bx * 2 + i < 7:
                    Y[bx * 2 + i] = total[i]

    result = analyze(kernel)
    p = result["modules"]["pipeline_overlap"]
    assert not p["unknown"], p.get("diagnostics")
    assert p["timing"] is not None
    reads = writes = 0
    for group in p["cta_work"]["groups"]:
        work = p["region_work"][group["iterations"]]
        reads += work["read_bytes"] * group["count"]
        writes += work["write_bytes"] * group["count"]
    expected = sum(
        4
        for bx in range(4)
        for k in range((min((bx + 1) * 3, 13) + 3) // 4)
        for i in range(2)
        for j in range(4)
        if bx * 2 + i < 7 and k * 4 + j < 13
    )
    assert reads == expected
    assert writes == 7 * 4


def test_data_dependent_guard_remains_unknown():
    @T.prim_func
    def kernel(X: T.Tensor((128,), "float32"), Y: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if X[i] > 0:
                    Y[i] = X[i]

    result = analyze(kernel)
    assert result["tile_cost"]["score"] is None
    assert result["modules"]["pipeline_overlap"]["diagnostics"][0]["code"] == "unsupported_scheduling"


@pytest.mark.parametrize("batches", [5, 1_000_000])
def test_two_axis_tails_preserve_cta_order_and_compress_batches(batches):
    @T.prim_func
    def kernel(X: T.Tensor((batches, 7, 13), "float32"), Y: T.Tensor((batches, 7, 13), "float32")):
        with T.Kernel(3, 4, batches, threads=128) as (bx, by, bz):
            tile = T.alloc_fragment((2, 5), "float32")
            T.copy(X[bz, by * 2 : by * 2 + 2, bx * 5 : bx * 5 + 5], tile)
            for i, j in T.Parallel(2, 5):
                if by * 2 + i < 7 and bx * 5 + j < 13:
                    Y[bz, by * 2 + i, bx * 5 + j] = tile[i, j]

    result = analyze(kernel)
    assert result["tile_cost"]["score"] is not None, result["diagnostics"]
    p = result["modules"]["pipeline_overlap"]
    distribution = p["cta_work"]
    assert distribution["repetitions"] == batches
    ordered = [p["region_work"][g["iterations"]] for g in distribution["groups"] for _ in range(g["count"])]
    expected = [sum(4 for i in range(2) for j in range(5) if by * 2 + i < 7 and bx * 5 + j < 13) for by in range(4) for bx in range(3)]
    assert [work["read_bytes"] for work in ordered] == expected
    assert [work["write_bytes"] for work in ordered] == expected
    assert sum(expected) == 7 * 13 * 4
    assert distribution["grid_blocks"] == 12 * batches


def test_independent_pipelines_drain_in_program_order():
    from tilelang.tiletune.ampere import schedule_cycles

    plan = dict(events=[dict(operations=[0], stage=0, async_group=0), dict(operations=[1], stage=1, async_group=-1)], max_stage=1)
    copy = dict(operation=0, bytes=64, first_consumer=1, last_consumer=1)
    rates = dict(
        global_bytes_per_cycle=16,
        async_copy_issue_bytes_per_cycle=64,
        async_copy_latency_cycles=20,
        copy_latency_cycles=0,
        barrier_cycles=2,
    )
    # Replay the tail independently: three iterations, last copy has 16 bytes.
    warp = service = 0
    ready = {}
    for step in range(4):
        if step < 3:
            size = 64 if step < 2 else 16
            warp += size / 64
            service = max(warp, service) + size / 16
            ready[step] = max(service, warp + 20)
        if step > 0:
            warp = max(warp, ready[step - 1]) + 2 + 10
    runs = [dict(count=2, copies=[copy], costs={0: 0, 1: 10}), dict(count=1, copies=[dict(copy, bytes=16)], costs={0: 0, 1: 10})]
    assert schedule_cycles(plan, [copy], {}, 3, rates, 1, runs=runs) == warp


def test_strided_scalar_access_is_not_a_dense_byte_count():
    @T.prim_func
    def kernel(X: T.Tensor((256,), "float32"), Y: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                Y[i] = X[2 * i]

    result = analyze(kernel)
    assert result["tile_cost"]["score"] is None
    assert any(d["code"] == "unresolved_memory_bounds" for d in result["diagnostics"])


def test_two_independent_native_pipelines_count_both_loops():
    @T.prim_func
    def kernel(A: T.Tensor((32, 48), "float16"), B: T.Tensor((48, 32), "float16"), C: T.Tensor((32, 32), "float16")):
        with T.Kernel(1, threads=128):
            left = T.alloc_shared((32, 16), "float16")
            right = T.alloc_shared((16, 32), "float16")
            accum = T.alloc_fragment((32, 32), "float32")
            T.clear(accum)
            for k in T.Pipelined(3, num_stages=2):
                T.copy(A[:, k * 16 : (k + 1) * 16], left)
                T.copy(B[k * 16 : (k + 1) * 16, :], right)
                T.gemm(left, right, accum)
            for k in T.Pipelined(2, num_stages=3):
                T.copy(A[:, k * 16 : (k + 1) * 16], left)
                T.copy(B[k * 16 : (k + 1) * 16, :], right)
                T.gemm(left, right, accum)
            T.copy(accum, C)

    before = kernel.script()
    result = analyze(kernel)
    assert kernel.script() == before
    p = result["modules"]["pipeline_overlap"]
    assert not p["unknown"], p["unknown"]
    assert result["tile_cost"]["score"] is not None
    loops = [n for n in p["region_schedule"]["variants"][0] if "kind" in n]
    assert [n["kind"] for n in loops] == ["pipeline", "pipeline"]
    assert [n["depth"] for n in loops] == [2, 3]
    assert [n["iterations"] for n in loops] == [3, 2]
    work = p["region_work"][0]
    visits = sum(1 for count in (3, 2) for _ in range(count))
    assert work["read_bytes"] == visits * (32 * 16 + 16 * 32) * 2
    assert work["write_bytes"] == 32 * 32 * 2
    assert work["gemm_flops"] == visits * 2 * 32 * 32 * 16

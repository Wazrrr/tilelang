import pytest
import tilelang.language as T
from tilelang.new_carver import analyze_prim_func, rank_records, CarverConfig
from test_analysis import gemm


LIMITS = {
    "sm_count": 132,
    "registers_per_sm": 65536,
    "shared_memory_per_sm": 233472,
    "shared_memory_per_block": 232448,
    "max_threads_per_sm": 2048,
    "max_threads_per_block": 1024,
    "max_blocks_per_sm": 32,
    "warp_size": 32,
}


@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (True, True)])
def test_gemm_tile_traffic_and_pipeline_storage(stages, trans_a, trans_b):
    result = analyze_prim_func(gemm(stages=stages, trans_a=trans_a, trans_b=trans_b), device_limits=LIMITS)
    cost = result["tile_cost"]
    assert cost["precision"] == "estimate"
    assert {r["buffer"] for r in cost["input_tiles"]} == {"A", "B"}
    assert cost["per_iteration_input_bytes"] == 2 * 32 * 32 * 2
    assert cost["input_bytes_per_block"] == 2 * 32 * 32 * 2 * 4
    assert cost["output_bytes_per_block"] == 32 * 32 * 4
    assert cost["shared_memory_bytes_estimate"] == 2 * 32 * 32 * 2 * max(1, stages)
    assert result["pressure"]["modeled_lower_bound"] == 8
    assert result["pressure"]["modeled_accumulator_registers_per_block"] == 1024
    assert cost["score"] == cost["traffic_bytes_per_block"] + 1  # one block, one wave
    assert any("k" in str(t["ranges"]) for t in cost["input_tiles"])


def test_dtype_and_repeated_tile_traffic():
    fp16 = analyze_prim_func(gemm(input_dtype="float16", acc_dtype="float16", extent=1), device_limits=LIMITS)["tile_cost"]
    fp64 = analyze_prim_func(gemm(input_dtype="float64", acc_dtype="float64", extent=1), device_limits=LIMITS)["tile_cost"]
    assert fp64["traffic_bytes_per_block"] == 4 * fp16["traffic_bytes_per_block"]
    assert fp64["shared_memory_bytes_estimate"] == 4 * fp16["shared_memory_bytes_estimate"]


def test_broadcast_uses_input_tile_not_output_elements():
    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, threads=32) as bx:
            tmp = T.alloc_fragment((32,), "float32")
            T.copy(A, tmp)
            for i, j in T.Parallel(32, 32):
                C[bx * 32 + i, j] = tmp[j]

    cost = analyze_prim_func(main)["tile_cost"]
    assert cost["input_bytes_per_block"] == 32 * 4
    assert cost["output_bytes_per_block"] == 32 * 32 * 4
    assert cost["grid_blocks"] == 2


def test_reduction_propagates_reduced_axis():
    @T.prim_func
    def main(A: T.Tensor((16, 32), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((16, 32), "float32")
            dst = T.alloc_fragment((16,), "float32")
            T.copy(A, src)
            T.reduce_sum(src, dst, dim=1)
            T.copy(dst, C)

    cost = analyze_prim_func(main)["tile_cost"]
    assert cost["input_bytes_per_block"] == 16 * 32 * 4
    assert cost["output_bytes_per_block"] == 16 * 4


def test_shared_and_thread_limits_control_waves():
    @T.prim_func
    def main(A: T.Tensor((1024, 32), "float16"), B: T.Tensor((32, 32), "float16"), C: T.Tensor((1024, 32), "float32")):
        with T.Kernel(32, threads=128) as bx:
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)
            T.copy(A[bx * 32 : bx * 32 + 32, :], a)
            T.copy(B, b)
            T.gemm(a, b, c)
            T.copy(c, C[bx * 32 : bx * 32 + 32, :])

    cost = analyze_prim_func(main, device_limits={**LIMITS, "sm_count": 2, "shared_memory_per_sm": 8192})["tile_cost"]
    assert cost["resident_blocks_per_sm_estimate"] == 2
    assert cost["num_waves_estimate"] == 8
    assert cost["traffic_bytes_grid_estimate"] == 32 * cost["traffic_bytes_per_block"]
    cost = analyze_prim_func(main, device_limits={**LIMITS, "sm_count": 2, "max_threads_per_sm": 128})["tile_cost"]
    assert cost["resident_blocks_per_sm_estimate"] == 1
    assert cost["num_waves_estimate"] == 16


def test_cost_errors_reach_the_caller(monkeypatch):
    def fail(*args):
        raise RuntimeError("cost failure")

    monkeypatch.setattr("tilelang.new_carver.cost.analyze_tile_cost", fail)
    with pytest.raises(RuntimeError, match="cost failure"):
        analyze_prim_func(gemm(), {"register_cap": 1})


def test_ranking_preserves_unknowns_ties_and_ignores_timings():
    records = [
        {"index": 3, "tile_cost": {"score": 20}},
        {"index": 2, "tile_cost": {"score": 10}},
        {"index": 1, "tile_cost": {"score": 10}},
        {"index": 4, "tile_cost": {"score": None}},
        {"index": 0, "tile_cost": {"score": 1}, "pre_lowering": {"would_reject": True}},
    ]
    ranked = rank_records(records)
    assert [r["index"] for r in ranked] == [1, 2, 3, 4, 0]
    assert ranked[0]["tie_first_rank"] == 1 and ranked[0]["tie_last_rank"] == 2
    for i, r in enumerate(records):
        r.update(latency_ms=100 - i, compiler_resources={"registers": 999}, status="benchmarked")
    assert rank_records(records) == ranked
    assert analyze_prim_func(gemm())["tile_cost"]["score"] is None  # absent device metadata
    assert CarverConfig(device_limits=LIMITS).to_cache_key_dict() != CarverConfig().to_cache_key_dict()

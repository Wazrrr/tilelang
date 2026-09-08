import pytest
import tilelang.language as T
from tilelang.tiletune import analyze_prim_func
from tilelang.transform import PassContext
from test_analysis import gemm
from test_cost import LIMITS


TARGET = {"kind": "cuda", "arch": "sm_90a"}


@pytest.mark.parametrize("threads,consumer_request", [(128, 240), (256, 240), (384, 160)])
def test_pipeline_policy_preserves_accumulator_and_accounts_for_producers(threads, consumer_request):
    results = [analyze_prim_func(gemm(stages=s, threads=threads), target=TARGET, device_limits=LIMITS) for s in range(4)]
    assert len({r["pressure"]["modeled_lower_bound"] for r in results}) == 1
    assert results[0]["pressure"]["warp_specialization"]["applies"] is False
    for result in results[1:]:
        policy = result["pressure"]["warp_specialization"]
        assert policy["status"] == "predicted", policy
        assert policy["launch_threads"] == threads + 128
        assert policy["consumer_register_request"] == consumer_request
        assert policy["compiler_registers_per_thread"] is None
        assert policy["register_reservation_per_block"] == 128 * 24 + threads * consumer_request
        cost = result["tile_cost"]
        assert cost["launch_threads"] == threads + 128
        assert cost["original_launch_threads"] == threads
        assert cost["resident_blocks_limits"]["registers"] == 65536 // policy["register_reservation_per_block"]


def test_policy_reads_function_settings_and_compile_override():
    func = gemm(stages=2).with_attr("tilelang_pass_configs", {"tl.disable_warp_specialized": True})
    assert analyze_prim_func(func, target=TARGET)["pressure"]["warp_specialization"]["status"] == "disabled"
    with PassContext(config={"tl.disable_warp_specialized": True}):
        result = analyze_prim_func(func, target=TARGET, pass_configs={"tl.disable_warp_specialized": False})
        assert result["pressure"]["warp_specialization"]["status"] == "predicted"
    assert (
        analyze_prim_func(gemm(stages=2), target={"kind": "cuda", "arch": "sm_80"})["pressure"]["warp_specialization"]["status"]
        == "not_applicable"
    )


def test_dtype_aware_bound_is_separate_from_policy_allocation():
    fp16, fp32 = [analyze_prim_func(gemm(stages=3, acc_dtype=d), target=TARGET)["pressure"] for d in ("float16", "float32")]
    assert fp32["modeled_lower_bound"] == 2 * fp16["modeled_lower_bound"]
    assert fp16["warp_specialization"]["register_reservation_per_block"] == fp32["warp_specialization"]["register_reservation_per_block"]


@pytest.mark.parametrize("predicated,disable_tma", [(True, False), (False, True)])
def test_unresolved_policy_does_not_reject(predicated, disable_tma):
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float16"), B: T.Tensor((128, 128), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, 32), "float16")
            b = T.alloc_shared((32, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.clear(c)
            for k in T.Pipelined(4, num_stages=2):
                if not predicated or k < 3:
                    T.copy(A[0, k * 32], a, disable_tma=disable_tma)
                    T.copy(B[k * 32, 0], b, disable_tma=disable_tma)
                    T.gemm(a, b, c)
            T.copy(c, C)

    result = analyze_prim_func(main, target=TARGET, device_limits=LIMITS)
    assert result["pressure"]["warp_specialization"]["status"] == "unknown"
    assert result["pressure"]["decision"]["keep"]
    assert result["tile_cost"]["score"] is None


def test_explicit_layout_policy_is_unknown_and_kept():
    result = analyze_prim_func(gemm(stages=2, explicit=True), target=TARGET, device_limits=LIMITS)
    assert result["pressure"]["warp_specialization"]["status"] == "unknown"
    assert result["pressure"]["decision"]["keep"]
    assert result["tile_cost"]["score"] is None

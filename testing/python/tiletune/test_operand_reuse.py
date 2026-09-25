"""Affine operand sharing, cache capacity, and distinct read/write service."""

from copy import deepcopy

import pytest
import tilelang.language as T

from test_ampere import analyze
from tiletune_core.operand_reuse import operand_read_service
from tiletune_core.region_schedule import estimate_region_cycles


def matrix(swizzle=False, *, nonaffine=False, extra_read=False):
    @T.prim_func
    def kernel(A: T.Tensor((256, 128), "float16"), B: T.Tensor((256, 128), "float16"), C: T.Tensor((256, 256), "float16")):
        with T.Kernel(4, 4, threads=128) as (bx, by):
            left = T.alloc_shared((64, 32), "float16")
            right = T.alloc_shared((64, 32), "float16")
            accum = T.alloc_fragment((64, 64), "float32")
            T.use_swizzle(panel_size=2, enable=swizzle)
            if extra_read:
                T.copy(A[0, 0], left)
            T.clear(accum)
            for k in T.Pipelined(4, num_stages=3):
                if nonaffine:
                    T.copy(A[(by * by % 4) * 64, k * 32], left)
                else:
                    T.copy(A[by * 64, k * 32], left)
                T.copy(B[bx * 64, k * 32], right)
                T.gemm(left, right, accum, transpose_B=True)
            T.copy(accum, C[by * 64, bx * 64])

    return kernel


@pytest.mark.parametrize("swizzle", [False, True])
def test_geometry_comes_from_ir_and_preserves_it(swizzle):
    func = matrix(swizzle)
    before = func.script()
    pipeline = analyze(func)["modules"]["pipeline_overlap"]
    geometry = pipeline["operand_reuse"]
    assert geometry["grid"] == [4, 4]
    assert [(tile["axes"], tile["bytes"]) for tile in geometry["tiles"]] == [([1], 4096), ([0], 4096)]
    assert geometry["buffer_depth"] == 3
    assert geometry["swizzle"] == (dict(pattern="rasterization2DRow", panel=2) if swizzle else None)
    assert func.script() == before


@pytest.mark.parametrize("options", [dict(nonaffine=True), dict(extra_read=True)])
def test_unproven_or_mixed_accesses_keep_per_cta_memory(options):
    pipeline = analyze(matrix(**options))["modules"]["pipeline_overlap"]
    assert "operand_reuse" not in pipeline


def geometry(swizzle=None):
    return dict(grid=[8, 4], tiles=[dict(axes=[1], bytes=64), dict(axes=[0], bytes=128)], buffer_depth=2, swizzle=swizzle)


def rates(capacity=4096):
    return dict(global_bytes_per_cycle=10, l2_bytes_per_cycle=40, l2_cache_bytes=capacity)


def test_reuse_follows_cuda_rasterization_and_conserves_unique_bytes():
    regular = operand_read_service(geometry(), rates(), 4)
    row = operand_read_service(geometry(dict(pattern="rasterization2DRow", panel=4)), rates(), 4)
    column = operand_read_service(geometry(dict(pattern="rasterization2DColumn", panel=2)), rates(), 4)
    assert regular["logical_bytes_per_iteration"] == row["logical_bytes_per_iteration"] == 768
    assert regular["unique_bytes_per_iteration"] == 576
    assert row["unique_bytes_per_iteration"] == column["unique_bytes_per_iteration"] == 384
    assert row["read_bytes_per_cycle"] == 20
    assert regular["read_bytes_per_cycle"] == pytest.approx(10 / 0.75)


def test_capacity_and_l2_throughput_bound_the_benefit():
    assert operand_read_service(geometry(), rates(100), 32)["read_bytes_per_cycle"] == 10
    full = operand_read_service(geometry(), rates(), 32)
    assert full["unique_bytes_per_iteration"] == 4 * 64 + 8 * 128
    assert full["read_bytes_per_cycle"] == 40
    assert operand_read_service(geometry(), rates(), 1)["read_bytes_per_cycle"] == 10
    assert operand_read_service(geometry(), {"global_bytes_per_cycle": 10}, 32) is None


def test_read_reuse_does_not_discount_output_writes():
    profile = dict(global_bytes_per_cycle=10, copy_latency_cycles=0, barrier_cycles=0)
    phase = dict(operation=0, work=dict(gemm_flops=0, shared_bytes=0, elementwise_ops=0, exp_ops=0, reduction_ops=0))
    node = dict(phase, active=True, external_work=dict(read_bytes=100, write_bytes=200, read_groups=0))
    pipeline = dict(unknown=[], performance_model=profile, phases=[phase], region_schedule=dict(variants=[[node]]))
    original = deepcopy(pipeline)
    assert estimate_region_cycles(pipeline)["cycles"] == 30
    pipeline["performance_model"] = dict(profile, global_read_bytes_per_cycle=20)
    assert estimate_region_cycles(pipeline)["cycles"] == 25
    assert original["performance_model"] == profile

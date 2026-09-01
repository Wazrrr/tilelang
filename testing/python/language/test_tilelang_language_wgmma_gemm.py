import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tilelang.layout import SwizzleMode
from tilelang.layout.swizzle import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)
from tilelang.cuda.language.intrinsics import (
    WGMMATensorCoreIntrinEmitter,
    WGSparseTensorCoreIntrinEmitter,
)
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import compute_gmma_descriptor


@pytest.mark.parametrize(
    "emitter_cls,k_arg",
    [
        (WGMMATensorCoreIntrinEmitter, "chunk"),
        (WGSparseTensorCoreIntrinEmitter, "warp_k"),
    ],
    ids=["dense", "sparse"],
)
@pytest.mark.parametrize("accum_dtype,accum_bits", [("float32", 32), ("float16", 16)])
@pytest.mark.parametrize(
    "block_row_warps,block_col_warps,warp_row_tiles,warp_col_tiles,expected_elements",
    [
        (4, 1, 16, 8, 4),
        (4, 1, 16, 256, 128),
        (8, 1, 16, 256, 128),
        (4, 2, 32, 128, 128),
        (8, 2, 16, 128, 64),
        (16, 1, 16, 64, 32),
    ],
)
def test_wgmma_accumulator_register_count_is_per_thread(
    emitter_cls,
    k_arg,
    accum_dtype,
    accum_bits,
    block_row_warps,
    block_col_warps,
    warp_row_tiles,
    warp_col_tiles,
    expected_elements,
):
    emitter = emitter_cls(
        a_dtype="float16",
        b_dtype="float16",
        accum_dtype=accum_dtype,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        **{k_arg: 32},
    )

    instruction_elements = (
        (4 * warp_row_tiles // emitter.wgmma_inst_m)
        * (warp_col_tiles // emitter.wgmma_inst_n)
        * emitter.wgmma_inst_m
        * emitter.wgmma_inst_n
        // 128
    )
    expected_regs = (expected_elements * accum_bits + 31) // 32

    assert instruction_elements == expected_elements
    assert emitter.wgmma_accum_regs == expected_regs


@pytest.mark.parametrize(
    "maker,mode,wgmma_field,sbo",
    [
        (make_full_bank_swizzled_layout, SwizzleMode.SWIZZLE_128B, 1, 64),
        (make_half_bank_swizzled_layout, SwizzleMode.SWIZZLE_64B, 2, 32),
        (make_quarter_bank_swizzled_layout, SwizzleMode.SWIZZLE_32B, 3, 16),
    ],
)
@pytest.mark.parametrize("n_dim,k_dim", [(64, 64), (64, 256), (128, 64), (64, 512)])
def test_compute_gmma_descriptor_k_major(maker, mode, wgmma_field, sbo, n_dim, k_dim):
    buf = tirx.decl_buffer((n_dim, k_dim), "float16", name="B", scope="shared")
    p = compute_gmma_descriptor(maker(buf), buf, transposed=False)
    assert p.swizzle_mode == mode
    assert p.swizzle_mode.wgmma_layout_type() == wgmma_field
    assert p.leading_byte_offset == 1
    assert p.stride_byte_offset == sbo


@pytest.mark.parametrize("k_dim,n_dim", [(64, 128), (64, 256), (64, 64)])
def test_compute_gmma_descriptor_mn_major_128b(k_dim, n_dim):
    buf = tirx.decl_buffer((k_dim, n_dim), "float16", name="B", scope="shared")
    p = compute_gmma_descriptor(make_full_bank_swizzled_layout(buf), buf, transposed=True)
    assert p.swizzle_mode == SwizzleMode.SWIZZLE_128B
    assert p.stride_byte_offset == 64
    # LBO is 0 for a single MN atom (n_dim == atom), else the multi-atom step.
    assert p.leading_byte_offset in (0, 512)


def test_compute_gmma_descriptor_same_layout_both_majors():
    # FlashMLA KV_shared [block_N, h_dim] (128B swizzle) must analyze as BOTH a
    # K-major operand (QK) and an MN-major operand (PV) -- the same physical bytes.
    buf = tirx.decl_buffer((64, 256), "float16", name="KV", scope="shared")
    lay = make_full_bank_swizzled_layout(buf)
    pk = compute_gmma_descriptor(lay, buf, transposed=False)
    pmn = compute_gmma_descriptor(lay, buf, transposed=True)
    assert pk.leading_byte_offset == 1 and pk.stride_byte_offset == 64
    assert pmn.swizzle_mode == SwizzleMode.SWIZZLE_128B


def _make_wgmma_kernel(gemm_op, *, m=64, n=64, k=16, accum_dtype=T.float16, threads=128):
    @T.prim_func
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((k, n), T.float16),
        D: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared((m, k), T.float16)
            B_shared = T.alloc_shared((k, n), T.float16)
            C_local = T.alloc_fragment((m, n), accum_dtype)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            gemm_op(A_shared, B_shared, C_local)
            T.wait_wgmma(0)
            T.copy(C_local, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize(
    "gemm_api",
    [T.wgmma_gemm],
)
def test_wgmma_gemm_has_no_implicit_wait(gemm_api):
    kernel = tilelang.compile(_make_wgmma_kernel(lambda A, B, C: gemm_api(A, B, C, clear_accum=True)), target="cuda")
    src = kernel.get_kernel_source()

    assert src.count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_dispatch_has_no_implicit_wait():
    kernel = tilelang.compile(
        _make_wgmma_kernel(lambda A, B, C: T.wgmma_gemm(A, B, C, clear_accum=True)),
        target="cuda",
    )

    assert kernel.get_kernel_source().count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_fence_count_is_independent_of_row_warpgroups():
    kernel = tilelang.compile(
        _make_wgmma_kernel(
            lambda A, B, C: T.wgmma_gemm(
                A,
                B,
                C,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=True,
            ),
            m=128,
            n=256,
            accum_dtype=T.float32,
            threads=256,
        ),
        target="cuda",
    )
    src = kernel.get_kernel_source()

    assert src.count("warpgroup_fence_operand") == 2
    assert ", 128);" in src
    assert ", 256);" not in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_rejects_mma_fallback():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((32, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float16)

            T.copy(A[0:32, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:32, 0:64])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.wgmma_gemm\(\) requires Hopper WGMMA lowering",
    ):
        tilelang.compile(main, target="cuda")


def _make_sliced_wgmma_kernel(M, N, K, num_k_tiles):
    k_tile = K // num_k_tiles

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        D: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.float16)
            B_shared = T.alloc_shared((N, K), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            for j in T.serial(num_k_tiles):
                T.wgmma_gemm(
                    A_shared[:, j * k_tile : (j + 1) * k_tile],
                    B_shared[:, j * k_tile : (j + 1) * k_tile],
                    C_local,
                    transpose_B=True,
                )
            T.wait_wgmma(0)
            T.copy(C_local, D)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_sliced_operand_emits_offset():
    # A sliced (non-zero-origin) operand must build the descriptor from the buffer
    # base and advance it with increase_descriptor_offset.
    kernel = tilelang.compile(_make_sliced_wgmma_kernel(64, 64, 64, 4), target="cuda")
    src = kernel.get_kernel_source()
    assert "increase_descriptor_offset" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize(
    "M,N,K,num_k_tiles",
    [
        (64, 64, 64, 4),
        (64, 64, 128, 4),
        (64, 64, 256, 8),
        (128, 64, 64, 2),
        (64, 128, 64, 2),
    ],
)
def test_wgmma_gemm_sliced_operand_correctness(M, N, K, num_k_tiles):
    import torch

    kernel = tilelang.compile(_make_sliced_wgmma_kernel(M, N, K, num_k_tiles), target="cuda")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    d = torch.empty(M, N, device="cuda", dtype=torch.float16)
    kernel(a, b, d)
    ref = (a.float() @ b.float().t()).half()
    torch.testing.assert_close(d, ref, rtol=1e-2, atol=1e-2)

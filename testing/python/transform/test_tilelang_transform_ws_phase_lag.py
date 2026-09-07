"""An unused producer warp must not miss reused mbarrier phases."""

from pathlib import Path
import subprocess
import sys

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing


def _run_delayed_producer(num_stages):
    import torch
    from tilelang.engine.callback import register_cuda_postproc_callback

    @T.prim_func
    def gemm(
        A: T.Tensor((512, 4096), "float8_e4m3fn"),
        B: T.Tensor((512, 4096), "float8_e4m3fn"),
        C: T.Tensor((512, 512), "float32"),
    ):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            a = T.alloc_shared((64, 32), "float8_e4m3fn")
            b = T.alloc_shared((64, 32), "float8_e4m3fn")
            c = T.alloc_fragment((64, 64), "float32")
            T.clear(c)
            for k in T.Pipelined(128, num_stages=num_stages):
                T.copy(A[by * 64, k * 32], a)
                T.copy(B[bx * 64, k * 32], b)
                T.gemm(a, b, c, transpose_B=True)
            T.copy(c, C[by * 64, bx * 64])

    @register_cuda_postproc_callback
    def delay_idle_producer(source, _target):
        # Delay one non-issuing producer warp in one CTA long enough for the
        # TMA leader and consumers to finish in the absence of a rendezvous.
        # This turns the scheduling-dependent FP8 hang into a reproducible one.
        marker = "tl::warpgroup_reg_dealloc<24>();"
        assert source.count(marker) == 1
        assert "tl::wgmma_ss" in source
        return source.replace(
            marker,
            marker
            + """
    if (threadIdx.x >= 96 && blockIdx.x == 0 && blockIdx.y == 0) {
      unsigned long long start = clock64();
      while (clock64() - start < 20000000ULL) { __nanosleep(100); }
    }
""",
        )

    tilelang.disable_cache()
    kernel = tilelang.compile(gemm, out_idx=[2], target={"kind": "cuda", "arch": "sm_90a"}, execution_backend="tvm_ffi")
    a = torch.ones((512, 4096), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.ones_like(a)
    out = kernel(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, 4096), rtol=0, atol=0)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_delayed_idle_producer_completes(num_stages):
    # A kernel hang cannot be recovered by an in-process Python timeout. Use a
    # child context so a regression terminates cleanly and releases the GPU.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--repro", str(num_stages)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--repro":
        _run_delayed_producer(int(sys.argv[2]))
    else:
        tilelang.testing.main()

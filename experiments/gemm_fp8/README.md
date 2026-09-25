# FP8 GEMM experiments

This family calls [`matmul.get_tir`](../../examples/gemm_fp8/example_tilelang_gemm_fp8.py)
directly. A=(M,K), B=(N,K), C=(M,N); inputs and output are FP8, with FP32
accumulation, transposed B, and the example’s direct fragment-to-global epilogue.
There is no scaling or FP16 fallback. Named cases use `float8_e4m3fn`; explicit
workloads also support `float8_e5m2`.

The five final cases cover 128-token continuous-batch decode, 1024-token prefill, FFN contraction,
a 4096-token projection, and a 4096-token FFN expansion. All five use E4M3 on
this branch. Training and validation
use the same serving dimensions at smaller token counts. Ampere is rejected
because it has no native FP8 tensor-core path; Hopper and Blackwell use the same
source kernel and configuration pool.

The 2,304 configurations retain all 288 schedules from
[`example_gemm_fp8_tiletune.py`](../../examples/gemm_fp8/example_gemm_fp8_tiletune.py):

| Parameter | Values |
| --- | --- |
| `block_M`, `block_N` | 32, 64, 96, 128, 192, 256 |
| `block_K` | 32, 64, 96, 128 |
| `num_stages` | 0, 1, 2, 3 |
| `threads` | 128, 256 |
| `enable_rasteration` | true, false |

All methods use the full ordered pool. Compiler and correctness failures remain
outcomes. Carver uses `FP8MatmulTemplate` with FP8 operands and FP32 accumulation.
TileTune requires a measured profile for the specific FP8 format and actual
matrix instruction, including the FP32-to-FP8 output conversion measured by
profile version 7; an FP16 profile cannot score FP8 work.
Profile version 8 additionally measures both MMA and WGMMA on Hopper so tiles
that select MMA use their own FP8 matrix rate. Regenerate older Hopper profiles
to cover those schedules.

The reference computes FP32 matmul and rounds the output to FP8. Checks require
the exact output dtype/shape, finite values, at most one FP8 quantization step
per element, and at most 2% relative norm error. The elementwise allowance covers
rounding midpoints reached from different FP32 accumulation orders; it does not
relax the checks for the other families.

```bash
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --plan
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper \
  --output experiments/results/gemm_fp8/hopper-v1
python -m experiments.gemm_fp8.census --device hopper --plan
python -m experiments.gemm_fp8.system.run --variant all --plan
```

Native CUDA FP8 needs SM89 or newer. A100 has no FP8 tensor instructions, so the
shared study records these cases as unsupported and continues other families.
Hopper/Blackwell measurements require those actual devices. HIP support is
restricted to the example’s OCP FP8 encoding on gfx950; FNUZ is a different
contract. Planning or cross-compilation alone is not GPU performance validation.

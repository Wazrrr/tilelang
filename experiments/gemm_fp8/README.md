# FP8 GEMM

This family calls `tl_gemm.get_tir` from
[`example_deepgemm_fp8_2xAcc.py`](../../examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py).
It preserves A=(M,K), pretransposed B=(N,K), uses E4M3 inputs, explicit FP32
scales, FP32 accumulation, and BF16 output. A scales are per row and K=128
block; B scales use a fixed 128x128 block layout.

The five final cases cover 256-token continuous-batch decode, 1024-token prefill, FFN contraction,
a 4096-token projection, and a 4096-token FFN expansion. All five use E4M3 on
this branch. Training and validation
use the same serving dimensions at disjoint token counts. This branch uses the
Hopper-specific source kernel and pool.

Training shapes (768,4096,4096) and (512,14336,4096), and validation shape
(3072,4096,4096), are disjoint from the test shapes.

The single expanded pool contains four native Hopper schedules:

| Parameter | Values |
| --- | --- |
| `block_M` | 64 |
| `block_N` | 16, 32, 64, 128 |
| `block_K` | 128 |
| `num_stages` | 4 |
| `threads` | 128 |

Candidate failures remain recorded. The native path requires CUDA SM89 or
later. The independent reference dequantizes both operands, performs FP32
matmul, and casts to BF16. Shape, dtype, finite-output, and numerical checks all
apply.

TileTune uses the lowered matrix instruction and exact operand/accumulator
dtypes to select measured primitive rates. Version-6 Hopper profiles include
both MMA and WGMMA measurements. Missing signatures stay unscored. Carver uses
the dedicated `FP8MatmulTemplate`; it retains the kernel dtype and lowers
E4M3FN to Carver's tensorizable E4M3 spelling internally.

```bash
python -m experiments.gemm_fp8.system.run --plan
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --plan
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --run-baselines
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --top-k 20
```

Baselines live under `results/<GPU>/baselines/`. See the
[shared protocol](../README.md) for monitoring, cache reuse and comparison.

# FP8 GEMM

This family calls `matmul.get_tir` from
[`example_tilelang_gemm_fp8.py`](../../examples/gemm_fp8/example_tilelang_gemm_fp8.py).
It preserves A=(M,K), pretransposed B=(N,K), FP32 accumulation, FP8 output,
rasterization and the example's shared-memory epilogue. Both `float8_e4m3fn`
and `float8_e5m2` are supported; final cases use `float8_e4m3fn`.

| Case | Development M=N=K | Final M=N=K |
| --- | ---: | ---: |
| `gemm_fp8_square` | 1024 | 4096 |
| `gemm_fp8_square_large` | 2048 | 8192 |

Training shapes (512,512,512) and (1024,256,768), and validation shape
(384,768,512), are disjoint from the test shapes.

The single expanded pool contains 2,304 schedules, including all 288 schedules
from `example_gemm_fp8_tiletune.py`:

| Parameter | Values |
| --- | --- |
| `block_M`, `block_N` | 32, 64, 96, 128, 192, 256 |
| `block_K` | 32, 64, 96, 128 |
| `num_stages` | 0, 1, 2, 3 |
| `threads` | 128, 256 |
| `enable_rasteration` | true, false |

Candidate failures remain recorded. FP8 requires CUDA SM89 or later; Ampere
records unsupported. Other backends require their own compilation and device
validation. The independent reference uses FP32 matmul before the FP8 cast.
Correctness requires at most one FP8 representable step per element and relative
energy error below 1e-3, as in the example. Shape, dtype and nonfinite output
checks also apply.

TileTune uses the lowered matrix instruction and exact operand/accumulator
dtypes to select measured primitive rates. Version-6 Hopper profiles include
both MMA and WGMMA measurements. Missing signatures stay unscored. Carver reuses
the existing `MatmulTemplate` and tensor-core policy with FP8 operands/output;
the E4M3FN spelling is registered in its CUDA precision table.

```bash
python -m experiments.gemm_fp8.system.run --plan
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --plan
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --run-baselines
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --top-k 20
```

Baselines live under `results/<GPU>/baselines/`. See the
[shared protocol](../README.md) for monitoring, cache reuse and comparison.

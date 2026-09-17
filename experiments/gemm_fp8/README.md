# FP8 GEMM

This family calls `matmul.get_tir` from
[`example_tilelang_gemm_fp8.py`](../../examples/gemm_fp8/example_tilelang_gemm_fp8.py).
It preserves A=(M,K), pretransposed B=(N,K), FP32 accumulation, FP8 output,
rasterization and the example's shared-memory epilogue. Both `float8_e4m3fn`
and `float8_e5m2` are supported; final cases use `float8_e4m3fn`.

The five final cases cover 128-token continuous-batch decode, 1024-token prefill, FFN contraction,
a 4096-token projection, and a 4096-token FFN expansion. All five use E4M3 on
this branch. Training and validation
use the same serving dimensions at smaller token counts. Ampere is rejected
because it has no native FP8 tensor-core path; Hopper and Blackwell use the same
source kernel and configuration pool.

Training shapes (32,4096,4096) and (512,14336,4096), and validation shape
(2048,4096,4096), are disjoint from the test shapes.

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

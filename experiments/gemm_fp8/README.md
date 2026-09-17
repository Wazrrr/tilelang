# FP8 GEMM experiments

This family calls `examples/gemm_fp8/example_tilelang_gemm_fp8.py` directly.
It measures `C = A @ B.T` with E4M3 inputs, FP32 accumulation and E4M3 output.
Its 2,304-entry configuration pool contains the example's
complete 288-entry grid and expands the tile dimensions to match the other
backend branches:

| Parameter | Values |
| --- | --- |
| `block_M`, `block_N` | 32, 64, 96, 128, 192, 256 |
| `block_K` | 32, 64, 96, 128 |
| `num_stages` | 0, 1, 2, 3 |
| `threads` | 128, 256 |
| `enable_rasteration` | true, false |

The five final cases cover 128-token continuous-batch decode, 1024-token prefill, FFN contraction,
a 4096-token projection, and a 4096-token FFN expansion. All development, final,
training and validation cases use E4M3 (`float8_e4m3fn`) and the same workload
names as the H200 study. Training and validation use the same serving dimensions
at smaller token counts. Ampere is rejected
because it has no native FP8 tensor-core path; Hopper and Blackwell use the same
source kernel and configuration pool.

Run the shared comparison with:

```bash
python -m experiments.gemm_fp8.tiletune.run --suite full --device blackwell \
  --output experiments/results/gemm-fp8
```

# FP8 GEMM experiments

This family calls `examples/gemm_fp8/example_tilelang_gemm_fp8.py` directly.
It measures `C = A @ B.T` with FP32 accumulation and FP8 output for both E4M3
and E5M2 inputs. The configuration pool is the example's complete 288-entry
grid: 3×3×2 tiles, four pipeline depths, two thread counts, and rasterization
on/off.

The final cases are 4096³ E4M3 and 8192³ E5M2. Training and validation use
separate shapes. Ampere is rejected because it has no native FP8 tensor-core
path; Hopper and Blackwell use the same source kernel and configuration pool.

Run the shared comparison with:

```bash
python -m experiments.gemm_fp8.tiletune.run --suite full --device blackwell \
  --output experiments/results/gemm-fp8
```

# FP8 GEMM experiments

This family calls the Blackwell-specific
`examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py` kernels.
It measures `C = A @ B.T` with E4M3 inputs, explicit packed UE8M0 scales at
K=128 granularity, FP32 accumulation, and BF16 output. The eight-entry pool
covers the TCGen05 two-CTA tiled kernel and persistent variants:

| Parameter | Values |
| --- | --- |
| `implementation` | `tcgen05_2cta`, `tcgen05_2cta_persistent` |
| `block_M`, `block_N`, `block_K` | 128, 256, 128 |
| `num_stages` | 6 |
| `threads` | 128 (tiled), 256 (persistent) |
| `group_size` | 1, 2, 4 |
| `store_block_N` | 64, 128 |
| `use_tma_store` | true, plus one group-4 non-TMA candidate |

The five final cases cover 256-token continuous-batch decode, 1024-token prefill, FFN contraction,
a 4096-token projection, and a 4096-token FFN expansion. All development, final,
training and validation cases use E4M3 (`float8_e4m3fn`) and the same workload
names as the H200 study. The M/N/K alignment is fixed for the two-CTA SM100
schedule. Hopper and Ampere branches use their own architecture-specific source
kernels and configuration pools.

Run the shared comparison with:

```bash
python -m experiments.gemm_fp8.tiletune.run --suite full --device blackwell \
  --output experiments/results/gemm-fp8
```

# FP8 GEMM experiments

This family calls
[`blockscaled_gemm.get_tir`](../../examples/gemm_fp8/example_mxfp8_blockscaled_gemm_a100.py)
directly. A=(M,K), B=(N,K), C=(M,N); global inputs use E4M3 storage with
explicit per-row, per-K128 FP32 scales. Since Ampere has no native FP8 tensor
instruction, tiles are converted to BF16 for tensor-core GEMM, accumulated in
FP32, and returned as BF16.

The five final cases cover 256-token continuous-batch decode, 1024-token prefill, FFN contraction,
a 4096-token projection, and a 4096-token FFN expansion. All five use E4M3 on
this branch. Training and validation
use the same serving dimensions at disjoint token counts. This is an
architecture-specific storage/compute emulation baseline, not native FP8 MMA.

The 72 configurations cover the Ampere BF16-compute schedule:

| Parameter | Values |
| --- | --- |
| `block_M`, `block_N` | 32, 64, 128 |
| `block_K` | 128 |
| `num_stages` | 0, 1, 2, 3 |
| `threads` | 128, 256 |

All methods use the full ordered pool. Compiler and correctness failures remain
outcomes. Carver models the BF16 compute path and conservatively accounts for
the explicit-scale operand traffic.

The reference dequantizes the E4M3 operands, computes FP32 matmul, and casts to
BF16. Checks require the exact BF16 output dtype/shape, finite values, and the
family numerical tolerance.

```bash
python -m experiments.gemm_fp8.tiletune.run --suite full --device ampere --plan
python -m experiments.gemm_fp8.tiletune.run --suite full --device ampere \
  --output experiments/results/gemm_fp8/ampere-v1
python -m experiments.gemm_fp8.census --device ampere --plan
python -m experiments.gemm_fp8.system.run --variant all --plan
```

A100 has no FP8 tensor instructions, so this branch measures E4M3 storage with
explicit scales and BF16 tensor-core compute. Hopper and Blackwell branches use
their native architecture-specific FP8 kernels. Planning or cross-compilation
alone is not GPU performance validation.

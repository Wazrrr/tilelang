# FP8 GEMM

All three CUDA branches call
[example_blockscaled_gemm.py](../../examples/gemm_fp8/example_blockscaled_gemm.py).
A=(M,K) and pretransposed B=(N,K) contain E4M3 values. Explicit FP32 scales have
shapes (M,K/128) and (N,K/128), independent of the candidate tile. Partial products
and the scaled total accumulate in FP32; the output is BF16.

A100 converts E4M3 values exactly to BF16 in shared memory because Ampere has no
native FP8 matrix instruction. H200/B200 use E4M3 shared operands. Input generation,
scale layout, mathematical reference, shapes and pool are identical. The native
choice is explicit in `experiments/backend.py`.

The single 576-config pool varies block_M/block_N over 32/64/96/128/192/256,
stages over 0/1/2/3/4/5/6/7 and threads over 128/256. block_K=128 is fixed by the scale layout. Final
shapes match BF16 GEMM, including small-M, square, rectangular and long-reduction
cases. Training/validation shapes remain separate from final holdouts.

Carver uses one traffic/wave adapter across architectures. It counts E4M3 global
loads, FP32 scale loads, BF16 output stores, actual shared-operand dtype and two
FP32 accumulator tiles. Scale arithmetic is reported without an invented latency.
The old three architecture-specific FP8 pools are retired from experiments.

See [the common contract](../BENCHMARK_CONTRACT.md) for complete shapes,
provenance and validation limits. For example:

```bash
python -m experiments.gemm_fp8.tiletune.run --suite full --device hopper --plan
```

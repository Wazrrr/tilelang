# Grouped GEMM experiments

This family calls [`grouped_gemm.get_tir`](../../examples/grouped_gemm/example_grouped_gemm_fwd.py)
with concatenated A/C rows and one B matrix per group. FP16/BF16 inputs use FP32
accumulation. The example owns its output index; the adapter preserves it.

The five development/final cases cover MoE decode, prefill, aligned up/down
projections and ragged groups. Exact shapes and independent two-training/one-
validation splits live in [`cases.py`](cases.py). The final cases are also frozen
in the common and grouped-family manifests.

The single ordered pool contains 576 configurations: `block_M=64`,
`block_N=16/32/48/64/80/96/112/128/160/192/224/256`, `block_K=16/32/48/64/96/128`,
`num_stages=0/1/2/3`, and `threads=128/256`. Fixing block_M keeps group sizes, offsets and padded
offsets identical for every candidate. Compilation and correctness failures
remain outcomes; all methods receive this same pool.

TileTune receives those actual integer vectors through `input_values` and checks
them against runtime tensors. This resolves masked group traffic without changing
the compiled program. Carver uses `GroupedMatmulTemplate` and its padded CTA
domain; its legacy score omits dispatch metadata traffic and rasterization.

```bash
python -m experiments.grouped_gemm.tiletune.run --suite full --device ampere --plan
python -m experiments.grouped_gemm.census --suite final --plan
python -m experiments.grouped_gemm.system.run --plan
```

Use the shared [comparison workflow](../README.md) for execution. Profiles and
measured results remain device-specific. See the
[unification audit](../backend_unification_20260917.md) for validation boundaries.

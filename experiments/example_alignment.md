# Example-kernel alignment (2026-09-16)

This records the initial alignment checkpoint. All four families now have one
complete expanded pool and no local/legacy experiment kernels: GEMM 2,304,
FlashAttention 320, KDA 720 and softmax 224 configs per case. See the
[current family contracts](README.md). Pool counts and measured results below
describe the earlier checkpoint, not the new space-version-5 pools.

The final eight cases now build their TileLang programs by calling the example
builders directly. This replaces the separate implementations selected by the
previous experiment pools. All final workloads remain FP16; program identity is
also tested at BF16.

| Family | Authoritative example builder | Previous mismatch | Current contract |
| --- | --- | --- | --- |
| GEMM | [`make_autotune_kernel_builder`](../examples/gemm/example_gemm_advanced_autotune.py) | The final runner used `kernels/tiled.py`: 3-D NN tensors, dummy bias and direct fragment-to-global output. The advanced replica existed but was only used by the legacy path. | 2-D A=(M,K), B=(N,K), C=(M,N), `transpose_B=True`, FP32 accumulation, shared-memory epilogue. |
| FlashAttention | [`flashattn`](../examples/flash_attention/example_mha_fwd_bshd.py) | Expanded candidates selected a rewrite with shared score/probability transfers and a full-KV pipelined causal loop. | The example's BSHD tensors, fragment probability path, FullRow GEMMs and causal loop bound. |
| KDA chunk output | [`tilelang_chunk_fwd_o`](../examples/kda/chunk_o.py) | Rewritten BHSD kernels, independent row/causal tiles, a different gated-query rounding sequence, and direct output. | BSHD tensors; hidden=(B,chunks,H,DK,DV); the example's two casts, GEMMs and shared-memory output; block_S=chunk_size. |
| Softmax | [`softmax_kernel`](../examples/online_softmax/online_softmax.py) | Separate full-row/streamed implementations and explicit layout choices. | The example's two passes and log2/exp2 recurrence, with a tail mask added in the example itself. |

The adapters in each family's `kernel.py` supply inputs, references, configuration
name mapping and output indices. They contain no replacement TileLang algorithm
for the final cases. Generic NN/fused/batched GEMM and recurrent KDA have since been retired
from the experiment interface. The former supplementary FP8/vector experiment workloads have since been retired.

## Why the shared-memory GEMM observation was real

`experiments/gemm/kernels/advanced.py` already contained a shared-memory epilogue,
but `experiments/gemm/kernel.py::make_case` selected `kernels/tiled.py` for the
final suite. That path copied its fragment directly to global C. The earlier
controlled comparison therefore measured different programs; adding shared C to
that old program improved 4096 cubed by about 8%, with little benefit at 8192
cubed. It did not show that the active final suite was already using TMA stores.
Both experiment interfaces now reuse the advanced example for plain NT GEMM.

The advanced and standard autotune GEMM examples store B as (N,K) and call
`T.gemm(..., transpose_B=True)`. The basic `examples/gemm/example_gemm.py`
stores B as (K,N). Final workloads explicitly record `transpose_b=True`; B is
prepared in the required layout outside kernel timing.

## Search-space changes

Space version 3 records the new pools. All 288 configurations from the advanced
GEMM example are protected before the 1,024-config cap, including its measured
H200 winner at both sizes:

```json
{"block_m":128,"block_n":256,"block_k":64,"stages":3,"threads":256,"warp_policy":"square","swizzle_panel":10}
```

Attention uses only the example's tile/stage/thread parameters and protects its
64x64 and 128x128 default launches. KDA protects all 90 native autotune configs
and varies only key/value tiles, stages and threads. Softmax varies its existing
row/column tiles and newly exposed launch-thread parameter. The old rewritten
attention/KDA/softmax knobs are rejected, not silently mapped to ignored values.

| Final H200 family | Current | Expanded | Large | Exhaustive |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 108 | 3,000 | 1,024 | 12,000 |
| Attention | 54 | 250 | 1,024 | 1,280 |
| KDA chunk output | 24 | 90 | 1,024 | 1,280 |
| Softmax | 6 | 32 | 224 | 224 |

These are declared candidate counts, before compilation/correctness checks.
The eight final large pools contain 6,592 candidates in total. No complete sweep
of these revised pools has been collected yet. Prior A100/H200 records remain
historical measurements of their original kernels, layouts, dtype and pools.
Their latencies and config indices cannot stand in for a new oracle.

## Minimal changes in the examples

- GEMM is importable as a package and exposes warp policy and swizzle panel with
  its original defaults (Square, panel 10). The optional missing static analyzer
  is imported only when static pruning is requested.
- KDA defers optional FLA comparison imports and random seeding to the executable
  test path. Its TileLang kernel body is unchanged.
- FlashAttention's example is unchanged.
- Softmax infers M/N from its input, exposes `threads=128`, and puts its demo under
  `main()`. Partial column tiles mask padding to negative infinity before max/sum
  reduction. Without that mask, zero-filled padding incorrectly contributes to
  the denominator for the 1,537-column final case. Aligned tiles keep the original
  recurrence. The experiment calls this same fixed example.

## Validation and recorded runs

- 16 structural equality checks: all eight final programs versus their example
  entry points, at FP16 and BF16, using the representative configs in
  `testing/python/experiments/test_example_kernels.py`.
- Four target-space checks confirm both GEMM final pools retain every one of the
  example's 288 schedules (including target-specific canonical aliases).
- 147 planning, protocol, family, learned-model and provenance regression checks.
- All eight FP16 final shapes compile and pass independent numerical references
  on H200; 24 additional GPU checks cover tails, masking and native schedules.
- All eight shapes also pass through the actual single-candidate brute-force
  runner, including compilation, correctness and event timing.

These runs used idle H200s, with one-second GPU/process monitoring. All accepted
runs recorded no foreign compute process; subsecond interference is not excluded.
The runner smoke check measured 728.5 TFLOPS at 4096 cubed and 718.5 TFLOPS at
8192 cubed with the protected example configuration in FP16. These are isolated
single-config checks, not newly tuned optima or a paired performance study.

Raw requests, logs, monitor records and summaries are under
[`results/h200-example-alignment-20260916/`](results/h200-example-alignment-20260916/summary.json).
Two initial softmax checks exposed an output-index adapter conflict; after
respecting the eager example's embedded output index, all eight reruns passed.
The initial failed attempts are retained separately from the accepted records.

Run/source fingerprints and XGBoost execution domains now include the actual
example sources, so changes to those builders invalidate current cached study
or model identities.

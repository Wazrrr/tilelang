# H200 FP8 GEMM

The experiment calls [the original example](../../examples/gemm_fp8/example_tilelang_gemm_fp8.py)
directly. A=(M,K) and B=(N,K) use E4M3; accumulation is FP32 and output C is E4M3.
Inputs and outputs match that example; there are no scale tensors.

The original [288-config grid](../../examples/gemm_fp8/example_gemm_fp8_tiletune.py)
is included in the expanded pool: M/N tiles 64/128/256, K tiles 32/64,
128/256 threads, and both rasterization settings. Stages extend from 0–3 to 0–7,
giving 576 candidates before compilation qualification. The original default
128x128x64, stages=3, threads=128, rasterization=False is included.

The adapter reference computes A@B.T in FP32 and casts to the example's output
dtype. Source fingerprints and measurement contract version 4 prevent reuse of
incompatible measurements. FP8 Carver support is deferred; the previous adapter
is not used for this operation.

CPU-only compilation qualification does not establish numerical correctness,
performance, or oracle retention. GPU experiments remain paused.

# Current workload and model contracts

The H200 suite contains BF16 GEMM, original E4M3 GEMM, grouped GEMM,
FlashAttention, and KDA token-parallel intra-chunk: five workloads per family.
See [the benchmark contract](BENCHMARK_CONTRACT.md).

TileTune's memory mode derives its score from the actual PrimFunc. Grouped GEMM
supplies immutable integer metadata through the generic `input_values` contract.
Unknown facts remain explicit; compiler success does not establish correctness
or oracle rank. Diagnostics are opt-in.

Carver work is deferred. KDA intra-chunk has no supported Carver adapter. The
previous FP8 adapter does not model the restored original FP8 operation, so it
is explicitly unavailable until updated. GEMM, attention and grouped GEMM
retain their existing adapters.

XGBoost training and measurements must carry current kernel source fingerprints
and pool identities. Results from incompatible contracts cannot be reused.

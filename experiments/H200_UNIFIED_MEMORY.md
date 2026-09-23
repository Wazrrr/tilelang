# Unified TileTune memory ranking on H200

The completed E2 suite provides a retrospective oracle-retention check for the
current 25 workload pools. It does not certify changed pools or kernels.

## Scoring

The memory scorer uses resolved memory accesses, loop visits, launch size, target
SM count, and pipeline depth extracted from the PrimFunc. It does not use kernel
names, family recognition, measured latency, oracle labels, or family weights.
Read-only integer inputs supply runtime metadata through the generic input contract.

```text
waves = ceil(grid_blocks / SM_count)
B = sum(access.bytes * access.visits) * waves
e = sum(access.visits for nonempty accesses)
E = e * waves
shortfall = max(0, 3 * SM_count - grid_blocks)
U = ceil(B * (grid_blocks + e + shortfall) / (grid_blocks + e))
D = IR pipeline depth (at least 1)
rank key = (U, -D, E)
score = 65535 * U * (U + 1) // 2 + (65535 - D) * (U + 1) + E
alpha budget = floor(alpha * pool_size)
selection = complete eligible score groups whose tail rank <= budget
```

This integer score expresses an ordering, not predicted latency or physical DRAM
traffic. Three SM waves are a fitted launch target, not an occupancy claim.
Per-CTA logical access count `e` dampens the underfill penalty for heavier CTAs.
Diagnostics such as register liveness and shared-memory lifetime are opt-in
through `memory_diagnostics=True`; they do not change memory scores. The
experiment explicitly selects `ranking_metric="memory", alpha=0.5`.

## E2 result

The CPU-only replay freezes every candidate score before joining E2 oracle
labels. With conservative equal-score tail ranks, `(U, -D, E)` retains all
25 workload oracles within 50%. The worst case is `gemm_fp8_prefill` at
272/576 (47.22%); `attention_short_causal` is 58/512 (11.33%). The three-wave
target was selected using these same E2 labels, so this demonstrates in-sample
fixed-pool fit only. It is not held-out validation or a new GPU measurement.

The replay is implemented by [e2_memory_formula_search.py](e2_memory_formula_search.py).

The post-compile resource policy remains a separate experiment policy. Its
historical calibration must be revalidated against current oracle configs.

See [the experiment contract](BENCHMARK_CONTRACT.md), [KDA](kda/README.md),
and [the sequential run plan](H200_THREE_RUN_PLAN.md). No GPU measurements are
performed by the CPU-only compilation qualification command.

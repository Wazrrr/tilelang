# Memory ranking

The completed E2 experiment contains measured oracles for the current 25 H200
workload pools. The CPU-only formula study freezes rankings from saved compiler
facts before loading those oracle indices. It evaluates conservative tail ranks,
so every configuration with the same primary score receives the end rank of its
tie group.

The selected production order is:

```text
B = logical byte-waves
e = logical global accesses per CTA
G = grid blocks
S = SM count
E = logical access-waves
D = pipeline depth

shortfall = max(0, 3*S - G)
U = ceil(B * (G + e + shortfall) / (G + e))
rank key = (U, -D, E)
```

This order retains all 25 E2 oracles within the strict 50% cutoff. The worst
rank is `gemm_fp8_prefill` at 272/576 (47.22%). The previous `(B, E, -D)` order
retains 24/25. Removing `e` from the adjustment retains 22/25.

The target of three SM waves was selected using these same E2 oracle labels.
The result therefore establishes in-sample fit for fixed kernels and pools; it
does not establish generalization or GPU speedup. See
[the formula replay](e2_memory_formula_search.py),
[the memory model](H200_UNIFIED_MEMORY.md), and
[the experiment contract](BENCHMARK_CONTRACT.md).

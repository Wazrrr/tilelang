# Unified TileTune memory ranking on H200

The current 25-workload suite has no complete oracle-retention validation yet.
Scoring every PrimFunc is not proof that the best measured configuration ranks
within the first 50%. Historical results do not certify changed pools or kernels.

## Scoring

The memory scorer uses resolved memory accesses, loop visits, launch size, target
SM count, and pipeline depth extracted from the PrimFunc. It does not use kernel
names, family recognition, measured latency, oracle labels, or family weights.
Read-only integer inputs supply runtime metadata through the generic input contract.

```text
waves = ceil(grid_blocks / SM_count)
B = sum(access.bytes * access.visits) * waves
E = sum(access.visits for nonempty accesses) * waves
D = IR pipeline depth (at least 1)
score = (B * (B + 1) // 2 + E) * 65536 + (65535 - D)
alpha budget = floor(alpha * pool_size)
selection = complete eligible score groups whose tail rank <= budget
```

This integer score expresses an ordering, not predicted latency or physical DRAM
traffic. Diagnostics such as register liveness and shared-memory lifetime are
opt-in through `memory_diagnostics=True`; they do not change memory scores.
The experiment explicitly selects `ranking_metric="memory", alpha=0.5`.

The post-compile resource policy remains a separate experiment policy. Its
historical calibration must be revalidated against current oracle configs.

See [the experiment contract](BENCHMARK_CONTRACT.md), [KDA](kda/README.md),
and [the sequential run plan](H200_THREE_RUN_PLAN.md). No GPU measurements are
performed by the CPU-only compilation qualification command.

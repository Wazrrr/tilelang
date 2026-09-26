# TileTune Core

`tiletune_core.memory.score_memory` provides a dependency-free ordering over
resolved logical global accesses, launch size, SM count, and IR pipeline depth.
The integer score keeps byte-waves dominant and uses pipeline depth only within
equal byte work, then uses logical access-waves to refine equal bytes and depth.
It exactly encodes `(byte_waves, -pipeline_depth, access_waves)` with integers.
`rank_records` assigns every equal primary score its group's tail rank,
and `select_top_k` retains an entire boundary tie by default. Its
`strict_budget=True` mode instead excludes a group that crosses the budget;
this preserves ties without ever selecting more than the requested count.
`alpha_budget` converts an original-pool fraction to that strict integer budget.
See the
[fixed-pool B200 replay](../experiments/MEMORY_RANKING.md).

Install independently of TileLang, TVM, PyTorch, and accelerator runtimes:

```bash
python -m pip install ./tiletune_core
```

The package has no runtime dependencies. TileLang wheels also include it.

`KernelFacts` is the versioned compiler boundary. Adapters resolve work counts,
ownership, storage, loop regions, and synchronization before exporting facts.
`BackendModel` supplies native engine service, allocation granularity, capacities,
and execution-unit residency. `AnalysisReport` records costs and structured
resource, policy, uncertainty, or unsupported diagnostics.

Native engine schedules use ordered, compressed regions. A dependency records
an operation ID and an iteration distance. An asynchronous operation releases
its issue engine before its completion latency elapses; consumers wait for its
completion. Repeated regions use max-plus exponentiation. The evaluator does
not parse expressions or infer compiler layouts.

The `ampere`, `hopper`, `blackwell`, `cdna4`, and `ascend910b` registrations define
engine/storage contracts. They **do not provide calibrated hardware profiles**.
Capacities, allocation units, and rates must be supplied explicitly. CDNA4 uses
`gfx950`; there is no CDNA3 profile fallback. Ascend exposes Cube, Vector, MTE,
L0, L1, and UB resources directly. Compiler adapters must account for resource
hierarchies (including CU/SIMD allocation) before reporting resolved peaks.

CUDA's existing numerical equations are preserved in the compatibility
evaluator `tiletune_core.cuda.evaluate_cuda_facts`. Its `cuda.v21` payload keeps
the established pipeline representation and resource estimates. The native
engine evaluator is a separate contract, not a claim that the current CUDA
compiler adapter supports native TCGEN05/TMEM or all non-CUDA schedules.

```python
import json
from tiletune_core import KernelFacts
from tiletune_core.cuda import evaluate_cuda_facts

facts = KernelFacts.from_dict(json.load(open("facts.json")))
report = evaluate_cuda_facts(facts)
print(report.score, report.units)
```

Compiler adapters must supply resolved facts using the versioned contract.
Automatic fact export and exploration depend on support in the compiler runtime.
`AttemptLedger` freezes attempts and retains failures without replacement.

## Lightweight rank fusion

`score_rank_product(accesses, grid_blocks, sm_count, pipeline_depth=1)` prepares
the original memory and launch-underfill scores from a shared access ledger.
Its `score` is intentionally `None`: the fused score depends on the candidate
pool. Supply its `component_scores` in each record's `tile_cost`, with
`ranking_metric="rank_product"`, to `rank_records`. The returned ranking entries
contain `score = tail_rank(memory) * tail_rank(underfill)` and
`component_tail_ranks`; the input records are not modified.

Only candidates with both components resolved and no existing rejection or
analysis failure participate in either view. Existing strict-budget selection
keeps whole product-score groups and never fills a shortfall by splitting a
boundary tie. No timing profile or compiler dependency is required.

## Fixed-rate max fusion

`score_work_max(accesses, compute, grid_blocks, sm_count, performance_model,
pipeline_depth=1)` accepts lean logical compute facts and fixed per-SM rates.
It sums compute service across matrix, scalar, exp, rsqrt and logical sum/max
reductions, multiplies by `ceil(grid_blocks / sm_count)`, and takes the maximum
with logical byte-waves divided by `global_bytes_per_cycle`. There is no
candidate-pool normalization, bound classification, residency gate or fitting.

For exact integer ranking, compute cycles are converted to memory-equivalent
bytes using the fixed bandwidth and rounded upward by less than one byte.
The score encodes `(max(byte_waves, compute_equivalent_bytes), -pipeline_depth,
access_waves)`, retaining whole ties and the original strict-budget policy.
Missing counts, nonzero-work rates, or mismatched matrix instruction/dtype
signatures produce an unknown score rather than a memory-only fallback.
`service_cycles` exposes both components; the encoded `score` is not a latency.
TCGen05 requires its own dtype-matched rate. Logical max reductions require
`reduction_max_ops_per_cycle`; sum reductions use `reduction_ops_per_cycle`.
These are logical primitive rates, not physical lane/shuffle work counts.

### Work-max/underfill rank product

`score_work_rank_product` takes the same arguments as `score_work_max` and
prepares two unchanged component scores: fixed-rate work-max and ungated
memory underfill. Pass its `component_scores` with
`ranking_metric="work_rank_product"` to `rank_records` to obtain:

```text
score = tail_rank(work_max) * tail_rank(underfill)
```

Both views use the same eligible pool. Equal component scores use the last
rank of the whole tie group, and equal products remain tied. The fused score
is recomputed per pool, not cached per kernel. Missing compute work or rates
leave the candidate unknown in both views; there is no memory-only fallback.
This opt-in heuristic does not change `work_max`, `rank_product`, eligibility,
the original-pool budget, or boundary-tie handling. It adds no fitted rate,
penalty coefficient, occupancy estimate, or compute/memory classification.

# TileTune code review guide

The [portable experiment guide](../../experiments/portable/README.md) describes
the version-18 extension to multiple workloads and targets. `targets.py` owns
offline target identity and model boundaries; `runtime.py` dispatches native
CUDA/HIP integration, while Ascend requires an external compiler worker.

TileTune reads an actual PrimFunc, propagates its required tiles, estimates
resources and timing, and ranks supplied configurations. Start with
[analysis.py](analysis.py) for the public boundary and [engine.py](engine.py)
for the complete stage order. The shared Python core is in `src/` within this
package; native operator metadata continues to come from the compiler.

## Source map

| Component | Responsibility |
| --- | --- |
| `analysis.py`, `config.py`, `__init__.py` | Public entry points, resolved settings, and exports |
| `targets.py` | Offline architecture/subgroup identity and explicit runtime detection |
| `engine.py` | Direct stage calls, family policy inputs, and report assembly |
| `src/ir.py`, `src/collector.py` | Region/operation records, native operator collection, and dependencies |
| `src/regions.py`, `src/propagation.py` | Symbolic geometry and one backward tile traversal |
| `src/buffer_facts.py`, `src/ir_utils.py` | Shared allocation facts, arithmetic, loops, and operation queries |
| `src/device.py` | Target capabilities and explicit device-capacity queries |
| `register_pressure.py` | Allocation descriptions, accumulator proof, budgets, and register policy |
| `tile_liveness.py` | Simultaneous local storage and loop-carried state |
| `global_memory.py`, `memory.py` | Timing-model traffic and family-independent logical memory work |
| `shared_memory.py` | Staged allocations, buffer lifetimes, and shared-storage reuse |
| `occupancy.py`, `warp_specialization.py` | Residency/waves and producer/consumer policy prediction |
| `compute.py` | Operation work, reduction ownership, and primitive service cycles |
| `schedule.py`, `pipeline.py`, `ranking.py` | Buffer recurrence, CTA dispatch, pipeline timing, and ranking |
| `families/` | GEMM/attention recognition, loop/operand roles, labels, and policy choices |
| `profiling/` | Fixed primitive kernels, reusable device measurements, and profile validation |
| `runtime.py`, `trace.py` | Autotuner integration, rejection enforcement, and intermediate snapshots |

## Shared flow

`analyze_prim_func()` validates the public inputs and resolves effective pass
settings once. `engine.analyze_kernel()` collects IR, propagates every captured
global output, and creates one buffer-fact map. These facts retain buffer
identity, shape, dtype, logical volume, and lazily evaluated layout properties.
Each stage keeps its own interpretation of ownership, packing, and uncertainty.

```text
analyze_prim_func
  → engine.analyze_kernel
      → src.collector._Collector
      → src.propagation._propagate_tiles
      → src.buffer_facts.collect_buffer_facts
      → register_pressure.analyze_register_pressure
      → engine.run_modules
          → families.select_specialization
          → tile_liveness.analyze_live_tiles
          → register_pressure.resolve_register_budget
          → warp_specialization.predict_warp_specialization
          → register_pressure.analyze_register_policy
          → global_memory.analyze_global_memory       [ranking enabled]
          → shared_memory.analyze_shared_memory       [ranking enabled]
          → occupancy.analyze_waves                   [ranking enabled]
          → pipeline.analyze_pipeline                 [ranking enabled]
          → ranking.apply_ranking_metric              [ranking enabled]
```

The engine calls shared stages directly. Families supply the selected loop,
operand roles, phase labels, external-access accounting mode, soft spill
allowance, and warp-specialization policy. They do not run separate register,
memory, or timing algorithms. Liveness receives an explicit loop; the engine
adds phase labels afterward. The numerical algorithms do not depend on those
labels.

Dependencies point from the entry points and engine to the stages and then to
`src/`. Core modules do not import the engine, runtime, or stage implementations.
Stages use core helpers directly rather than importing private helpers from
the public `analysis.py` module. The separate `propagate_inputs()` API uses the
same collector and propagation algorithm without storage or timing analysis.

## Follow one GEMM

Use [the trace example](../../examples/gemm/example_gemm_tiletune_trace.py).
It builds a 256×256×256 GEMM with 64×64 output tiles, a reduction tile of 32,
128 consumer threads, two stages, FP16 inputs, and an FP32 output. Its device
limits and primitive rates are illustrative constants.

1. The collector captures A/B copies, the GEMM accumulator, the final store,
   eight reduction iterations, and sixteen launched blocks. Propagation follows
   each required C tile back to its A/B regions.
2. Register analysis establishes the accumulator bound. Liveness estimates
   simultaneous local storage using the same buffer facts. Supported Hopper
   policy adds 128 producer threads and reserves 33,792 registers per block.
3. Global-memory analysis charges 4,096 bytes each for A and B on every one of
   eight iterations, plus one 16,384-byte C store: 81,920 bytes per block.
4. Shared-memory analysis assigns two stage copies to each 4,096-byte input
   buffer. Their lifetimes overlap, giving 16,384 bytes with no storage reuse.
5. Occupancy uses the minimum of thread, shared-memory, register, and block
   limits. Registers allow one block per SM; the example's four SMs process
   sixteen blocks in four waves.
6. `traffic_waves` gives `(81,920 + 1) × 4 = 327,684`. The example selects
   `pipeline_time`, which uses `compute.py` for operation service and
   `schedule.py` for per-buffer readiness/reuse. Uniform grid time is the CTA
   duration under modeled contention multiplied by four waves.

The completed memory report combines global traffic and shared storage under
the existing `modules.memory_traffic` key. Existing `waves`, register-policy,
pipeline, ranking, and trace fields retain their meanings.

## Register and timing contracts

`register_pressure.py` describes allocations and establishes an accumulator
lower bound before family recognition. `tile_liveness.py` estimates the peak
sum of overlapping local buffers and preserves carried state across the supplied
loop. These results share facts but retain different guarantees: only proven
demand and resolved physical constraints justify resource rejection.

The existing liveness estimate balances storage over launch threads. Explicit
layout ownership can therefore establish a higher per-thread accumulator bound.
This reorganization preserves that behavior; correcting ownership in the broader
liveness model is a separate algorithm change.

`analyze_register_policy()` returns a decision. The engine finishes analysis;
`TileTuneSession` later enforces `keep=False` before lowering. `report_only`
records violations without enforcing this gate. A positive `top_k` still
excludes unscored and pressure-rejected entries and freezes selection before
compilation. By default, selection includes the complete score group crossing
`top_k`. Set `strict_top_k=True` to treat `top_k` as a hard budget: TileTune
keeps a boundary group only when its conservative tail rank is at most the
budget, so it never splits a tie and may select fewer than `top_k` candidates.
`alpha` is a mutually exclusive first-class alternative: it uses
`floor(alpha * original_pool_size)`, automatically enables the strict tie
policy, counts failures and unknowns in the denominator, and selects only
eligible scored candidates. Failed selected candidates are never replaced.

For timing, read `compute.estimate_phase_cycles`,
`schedule.buffer_transition`, `pipeline.estimate_pipeline_cycles`, then
`ranking.apply_ranking_metric`. Ranking recomputes timing with modeled CTA
contention; the pipeline checkpoint also contains a single-CTA timing view.
Missing rates or unsupported schedules remain unknown. Unexpected stage errors
propagate; the autotuner records `analysis_failed` and stops that candidate.

## Imports, profiling, and validation

Package-root exports, settings, signatures, reports, and trace checkpoints are
stable for the existing timing metrics. Internal imports use the source map
above; obsolete forwarding modules are removed. Analysis version 39 disables
the bound-aware occupancy gate while preserving launch-underfill ranking,
B300 support, native alpha, lean diagnostics, and the declared metadata contract;
device-profile version 4 is unchanged.

`profiling/device_profile.py` and `profiling/device_probes.py` moved together
without content changes. Their source fingerprints are unchanged. Loading a
profile remains an offline operation, and hardware probing requires an explicit
`profile_device()` call. `profiling/profile_schema.py` performs validation without
querying or executing on a device.

Existing tests in [testing/python/tiletune](../../testing/python/tiletune) cover
propagation, ownership, shared storage, timing, uncertainty, traces, top-K, and
autotuner integration. The refactor also compares complete offline reports across
GEMM dtypes, stage counts, attention layouts, and both ranking metrics.

## Follow the data in a trace log

Set `trace_path` to append readable intermediate snapshots:

```python
result = analyze_prim_func(
    func,
    TileTuneConfig(trace_path="/tmp/tiletune_trace.log", ranking_metric="pipeline_time", performance_model=profile),
    target=target,
    device_limits=limits,
)

# The autotuner labels each analysis block with its config index and arguments.
tuner.set_tiletune_args(True, trace_path="/tmp/tiletune_trace.log")
```

The same option works in the `tiletune={...}` dictionary on `@tilelang.autotune`.
Tracing requests a fresh tuner run even if a tuning result is cached. The log
path does not change the kernel/config cache identity. Tracing adds serialization
and file I/O, so leave it off when measuring analysis overhead.

Each block includes a unique analysis ID, numbered checkpoints and the source
file/line that emitted them. Checkpoints use the variable names from the code:

| Checkpoint | What to inspect |
| --- | --- |
| `inputs`, `prim_func` | Target, config, supplied hardware inputs, candidate label, and original IR |
| `col` | Buffers, scopes, shapes, dtypes, native metadata, read/write regions, dependency indices, loops and threads |
| `tile_propagation` | The single backward traversal: per-operation demands, input tiles and loop coverage with symbolic launch offsets |
| `pressure.accumulator` | Initial accumulator lower bound using the same tile demands |
| `specialization` | Recognized family, operand roles and loop |
| `pressure.tile_liveness` | Live register buffers per operation and peak estimated storage |
| `pressure.warp_specialization` | Predicted thread partition, producer classification and register requests |
| `pressure.register_policy` | Physical reservation, logical demand, allowance and rejection decision |
| `memory`, `waves` | Tile bytes, shared storage, occupancy limits and grid waves |
| `pipeline` | Each operation's work, producer-buffer consumers, profile rates and single-CTA timing |
| `ranking` | Timing recomputed with active CTA contention, grid timing, and final score; or disabled/unknown status |
| `tile_cost` | Combined cost report returned to the caller |

Follow `buffer_id` across regions, GEMM operands and live sets, and follow
`operation`/`index` through dependencies, phase work and phase timing. Bounds and
other symbolic IR expressions are printed as strings; `null` retains unknown
values. Snapshots are serialized at the checkpoint, so the initial `col` does
not later acquire demands added by propagation. Report-only fields added after
a checkpoint, such as a module's `implementation`, appear in the final report.

Blocks append in completion order, with each analysis kept together across
compilation threads in the same process. Use separate paths for separate worker
processes. Analysis exceptions flush earlier checkpoints; file or formatting
errors are logged without changing analysis decisions.

For a small example that requires no GPU execution:

```bash
python -m examples.gemm.example_gemm_tiletune_trace --output /tmp/gemm_trace.log
```

The [example](../../examples/gemm/example_gemm_tiletune_trace.py) traces an
actual GEMM PrimFunc with explicitly illustrative device limits and primitive
rates. Use your measured `performance_model` for performance interpretation.

## Family-independent memory ranking

`TileTuneConfig(ranking_metric="memory", ...)` analyzes the supplied PrimFunc
through common operator regions, loop visits, dependencies, storage facts, and
backend inputs. This path does not recognize a whole kernel as GEMM, attention,
softmax, or another family. It bypasses family selection, compute profiles,
pipeline timing, warp-specialization prediction, and occupancy prediction.
Explicit family hints are rejected so a new kernel cannot silently depend on a
kernel-specific scoring adapter.

The score is an exact lexicographic integer encoding of logical global
byte-waves, the pipeline depth already present in the IR, and logical access
waves:

```text
logical_byte_waves = sum(access.bytes * access.visits)
                   * ceil(grid_blocks / SM_count)
logical_access_waves = sum(access.visits for nonempty accesses)
                     * ceil(grid_blocks / SM_count)
score = 65535 * B * (B + 1) // 2
      + (65535 - pipeline_depth) * (B + 1)
      + E
where B = logical_byte_waves and E = logical_access_waves
```

Byte work therefore always dominates: pipeline depth only orders candidates
with identical byte-waves, then fewer logical requests order equal bytes and
depth. The summed band widths are exact because `0 <= E <= B`; no floating-point
score or arbitrary event bound is used. Original index orders only identical
triples. Every equal-score member receives the group's last position as its
conservative rank, and default top-K retains the whole boundary group.

### Bound-aware memory ranking

`ranking_metric="bound_aware"` keeps the family-independent memory path and
adds a launch-underfill adjustment. The compute/memory occupancy gate is
disabled: classification is diagnostic only and cannot change the score.
The diagnostic divides dynamic tensor-core FLOPs by distinct global tensor
bytes and compares that arithmetic intensity with the target architecture's
reference ridge point. The current table uses `200 FLOPs/byte`
for Ampere and `281.25 FLOPs/byte` for B200 (`sm_100a`) and B300
(`sm_103`, `sm_103a`). Both use the same coarse dense BF16/FP16 reference:
2.25 PFLOP/s divided by 8 TB/s of HBM3e bandwidth. NVIDIA's HGX table lists
36 sparse BF16/FP16 PFLOP/s across eight B300 GPUs; dense performance is half
that figure. The reference sources are the NVIDIA HGX specification table
(`https://www.nvidia.com/en-us/data-center/hgx/`) and Blackwell Ultra architecture
overview (`https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/`).
This is not a dtype-specific calibrated performance model. Other Blackwell
architectures do not inherit the B300 reference. Every classification, including
an unknown architecture or unresolved kernel, retains an occupancy penalty of
one and receives the launch-underfill adjustment when `bound_aware` is selected.

This path no longer estimates resident warps or performs additional shared-memory
analysis for an occupancy penalty, even for compute-bound kernels. It does not
call the timing, full occupancy, register-allocation, warp-specialization, or
family models. The three-level lexicographic key is:

```text
W = ceil(grid_blocks / SM_count)
Q = max(0, 3 * SM_count - grid_blocks)
U = ceil(logical_bytes_per_CTA * W
         * (grid_blocks + accesses_per_CTA + Q)
         / (grid_blocks + accesses_per_CTA))
key = (U, -pipeline_depth, logical_access_waves)
```

The three-SM-wave target detects underfilled launches; the per-CTA logical
access count dampens its penalty. No separate launch-wave component is added:
`U` already contains `W`. The diagnostic split and ridge point are reported
under `modules.bound`, together with `occupancy_gate_enabled=false` and
`occupancy_penalty=1`; resident-warps facts are no longer estimated by this path.
The launch target and shortfall are reported under `modules.ranking`. Exported
facts use `bound_aware.v1`.

The metric remains opt-in; the default is still `memory`. The portable runner
accepts the `b300` target preset and `--metric bound_aware` without requiring a
primitive timing profile:

```bash
python -m experiments.common.run --devices b300 --workloads gemm_decode \
  --method analyze --metric bound_aware --output experiments/results/b300-bound-aware
```

### Pool-scoped rank product

`ranking_metric="rank_product"` combines the original memory order and its
launch-underfill variant without adding timing profiles, compute/memory
classification, occupancy estimation, or hardware-specific weights:

```text
score = tail_rank(memory) * tail_rank(underfill)
```

Both component ranks use the same eligible candidate pool. Equal component
scores share their group's last rank, and equal products also form complete tie
groups. `alpha=0.5` keeps the existing original-pool budget and excludes a whole
group that crosses it; failed, rejected, or incompletely scored candidates do
not enter either component ranking.

Individual analyses cache only `tile_cost.component_scores`, with
`score_scope="candidate_pool"` and `score=None`. `rank_records` computes the
product after the pool is known and reports it, together with
`component_tail_ranks`, in each ranking entry. Changing the pool recomputes the
ranks; cached kernel facts never contain a reusable fused score. Exported
access-ledger facts use `rank_product.v1` and can be replayed with
`tiletune_core.score_rank_product` followed by `rank_records`.

```bash
python -m experiments.common.run --devices b300 --workloads gemm_decode \
  --method top_k --metric rank_product --alpha 0.5 --output experiments/results/b300-rank-product
```

### Fixed-rate memory/compute max (opt-in)

`ranking_metric="work_max"` keeps the lean, family-independent analysis path
and requires an explicit `performance_model` (or dtype-specific CLI profiles).
It does not replace the default `memory` metric:

```text
waves = ceil(grid_blocks / sm_count)
memory_cycles = logical_bytes_per_cta * waves / global_bytes_per_cycle
compute_cycles = waves * sum(logical_work_per_cta[kind] / rate[kind])
service = max(memory_cycles, compute_cycles)
```

All rates use logical work per SM per cycle. No candidate-pool min/max,
candidate timings, occupancy gate, or kernel-family policy enters the score.
Compute counting includes matrix FLOPs, scalar value arithmetic, exp/exp2,
rsqrt and sum/max reductions. Parallel and serial visits are counted once;
buffer-index arithmetic is excluded. Unsupported computation or missing
nonzero-work rates leaves the score unknown, never silently zero.

The scorer converts compute cycles to equivalent bytes at the fixed bandwidth,
rounds upward by less than one byte, and preserves the existing ordering by
deeper pipeline and fewer accesses only within equal normalized work. Equal
complete keys stay tied; `alpha=0.5` still excludes entire boundary groups and
never exceeds half the original pool. Resource rejection thresholds do not change.

Use independently measured primitive rates with matching target, matrix dtype
and instruction. TCGen05 cannot borrow an MMA/WGMMA rate; max reductions need
`reduction_max_ops_per_cycle` from a regenerated profile. `memory_regime` is
explicit: logical byte requests are not measured HBM traffic. The model omits
cache-capacity effects, physical scalar ownership, synchronization, shared/TMEM
service and block-scale/two-CTA overhead; it is a ranking proxy, not a latency
prediction. Profiles are never measured by analysis itself.

`modules.compute_work` records the compute ledger, and
`modules.ranking.service_cycles` exposes both normalized views. Portable facts
use `work_max.v1` and replay through `tiletune_core.score_work_max` without TVM.
The portable CLI accepts `--metric work_max` and keeps profile loading enabled.

This metric is experimental and opt-in. The default remains `memory`; resource
policies and candidate pools are unchanged. Fusion is an ordinal pruning
heuristic, not a latency estimate or a guarantee of oracle retention.

### Work-max/underfill rank product

`ranking_metric="work_rank_product"` combines the unchanged `work_max` order
with the unchanged ungated underfill order:

```text
score = tail_rank(work_max) * tail_rank(underfill)
```

It uses the same compute facts and fixed primitive profiles as `work_max`.
There is no extra penalty weight, pool-max cost normalization, bound gate,
occupancy predictor, or kernel-family rule. Each component preserves its
`(-pipeline_depth, access_waves)` ordering; complete equal-product groups
remain tied under the existing strict original-pool budget.

Select it with `--metric work_rank_product` and the existing dtype-specific
profile configuration. Portable facts use `work_rank_product.v1` and replay
through `tiletune_core.score_work_rank_product` followed by `rank_records`.
Missing nonzero-work rates make both ranking views unknown, never silently
falling back to memory. Analysis version 42 separates cached reports.

The default remains `memory`, and the existing `work_max` and `rank_product`
metrics are unchanged. Exact-best retention on a development ledger is not
a guarantee for unseen workloads or new measurements.

The frozen E1/E2/E3 coordinator accepts `--gpu-model B300`, verifying CUDA name,
compute capability and physical UUID even if NVML reports an alias. This device
option does not change the frozen protocol's default `memory` metric. Its B200
spill/local thresholds transfer unchanged to B300 and are independent of the
ranking metric; they are not a guarantee that every B300 oracle is retained.

This is logical work rather than measured traffic: it does not model cache
behavior, coalescing, bandwidth, transaction size, compute throughput, or
physical occupancy. Opaque operations that may access a global buffer produce
an explicit unknown score. Opaque control, barrier, and other local operations
do not prevent known global accesses from being scored. A lowered `While` is
counted only when its scalar initialization, constant limit, unconditional
positive increment, and conservative maximum visits are provable. Manual
pipeline depth can also be read from an asynchronous global-to-shared copy
whose leading shared-buffer axis is indexed modulo that axis's extent. Other
dynamic loops remain unknown and therefore cannot silently prune a candidate.

When `input_values` supplies verified read-only one-dimensional integer
metadata, memory analysis substitutes those values in its private analysis
view. Lean mode resolves lookup indices, loop bounds, and scored extents while
deferring complete address and predicate simplification; uncertainty retries
the eager path. `ir_context.metadata_resolution` reports `deferred`, `eager`,
or `not_needed`. The original PrimFunc remains unchanged for compilation.

The direct softmax test in
[test_memory.py](../../testing/python/tiletune/test_memory.py) supplies a new
PrimFunc with masked loads/stores and two reductions. The test disables family
dispatch and timing/occupancy policy calls, checks that the IR is unchanged,
and replays the exported `memory.v2` facts through the dependency-free core.
That verifies the direct-analysis boundary for softmax; it is not a claim that
all future opaque primitives or performance orderings are already supported.

On the frozen B200 five-family pool, all 25 oracle winners have conservative
tail rank within 50%; 18 are within 20%. The completed live audit used a strict
user-supplied `alpha` of 50%, so a boundary score group was excluded rather than
split or expanded past the budget. All 25 oracle guards pass; the worst live
oracle tail rank is 31.25%. See
[the replay and GPU methodology](../../experiments/MEMORY_RANKING.md).

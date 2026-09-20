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

## Memory ranking and optional diagnostics

`TileTuneConfig(ranking_metric="memory", alpha=0.5)` uses a lean analysis by
default. It reads global accesses, loop visits, launch size and pipeline depth
from the PrimFunc, then orders candidates by logical byte work, request count
and depth. The [H200 study](../../experiments/H200_UNIFIED_MEMORY.md) describes
the score and its fixed-pool results.

Set `memory_diagnostics=True` to additionally construct reaching dependencies,
report backward tile propagation, estimate live register tiles, and plan shared
storage from tile lifetimes. These diagnostics do not change the memory score,
resource rejection, or candidate selection. Their default reports are explicitly
disabled; dependency lists are `null`, not empty graphs. `trace_path` and
`facts_path` do not implicitly enable these analyses.

Launch-limit checks always run. Backward propagation still runs internally when
dense MMA accumulator demand can prove a rejection under a strict register
policy. Compiler resource checks remain in the runtime after lowering. The
legacy `pipeline_time` and `traffic_waves` modes retain their full analyses.

When `input_values` supplies integer metadata, lean memory analysis resolves
metadata lookup indices, loop bounds, and access extents, while deferring
simplification of complete addresses and predicates. Metadata loads still count
toward memory work, and lookup bounds and read-only checks remain enabled. If
deferred collection reports uncertainty, collection retries with eager
simplification. This rule is shared by all kernels. Diagnostics and timing modes
use eager resolution; `ir_context.metadata_resolution` reports `deferred`,
`eager`, or `not_needed`.

```python
# Default: lean memory analysis and strict 50% selection.
tuner.set_tiletune_args(True, ranking_metric="memory", alpha=0.5)

# Opt in when inspecting dependencies and storage lifetimes.
tuner.set_tiletune_args(True, ranking_metric="memory", alpha=0.5, memory_diagnostics=True)
```

Analysis version 37 includes compiler-only resource policy in the cache identity.
Deferred metadata resolution was introduced in version 36. The diagnostic
setting remains part of the cache identity.
Portable memory facts use `memory.v3`, whose `dependencies` field may be `null`;
the score inputs and formula are unchanged from `memory.v2`.

## Post-compile resource policy

`post_compile_policy` overrides `mode`, `register_cap`, `max_spill_bytes`, and
`max_local_bytes` only for compiler-resource checks. Omitted entries inherit
the ordinary settings. With no override, the existing behavior is preserved.
The override cannot change memory scoring, pre-lowering eligibility, or alpha
selection. It is included in cache identity and the effective limits appear in
each `post_compile.policy` report.

```python
tuner.set_tiletune_args(
    True,
    ranking_metric="memory",
    alpha=0.5,
    mode="report_only",
    max_spill_bytes=None,
    max_local_bytes=None,
    post_compile_policy={
        "mode": "reject",
        "max_spill_bytes": 64,
        "max_local_bytes": 64,
    },
)
```

This example permits up to 64 bytes in each PTXAS spill counter and 64 local
bytes, while retaining hardware register, launch, and shared-memory checks.
These are compiler counters, not measured runtime traffic. Missing counters
remain unknown rather than being interpreted as zero. Rejection happens after
compilation and before benchmarking; rejected selections are not replaced.

The H200 memory experiment runner supplies explicit family budgets from
[resource_policy.py](../../experiments/common/resource_policy.py): 64 bytes for
attention, 128 for FP8 GEMM, and zero for the other experiment families. This
policy uses the workload declaration outside the analyzer; memory scoring
remains family-independent. See the [oracle resource audit](../../experiments/H200_UNIFIED_MEMORY.md).

## Grouped compilation and recovery

`tuner.run(use_pipeline=True, enable_grouped_compile=True, group_compile_size=8)`
compiles up to eight selected configurations together when their effective
compiler settings agree. Selection remains frozen before compilation.

Per-config elaboration/lowering errors and resource rejections affect only that
config. If a shared device or host build fails, the compiler splits unfinished
configs into smaller groups, down to singletons. It reuses lowered IR, preserves
effective options and post-compile checks, and does not retry rejected configs
or refill the selection. Valid neighbors can still compile and benchmark.

Affected TileTune records include `grouped_compile_fallbacks` with the failed
group, retry groups and error. Device/host compile costs accumulate all attempts,
including failed shared builds. The timing log also retains each build attempt.

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
| `global_memory.py` | Logical external traffic per tile visit |
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
compilation. Failed selected candidates are never replaced.

Equal primary scores share their group's last rank. Runtime selection retains
the complete group at the K boundary, so `selected_count` may exceed requested
K; `budget_excess` records the difference. `position` is only the deterministic
report order. Offline strict-budget comparisons require the whole group to fit
within K before counting its oracle hit.

`TileTuneConfig(alpha=0.5)` selects complete score groups within
`floor(0.5 * original_pool_size)`. Failed and unknown candidates stay in that
denominator. Alpha is mutually exclusive with `top_k` and exploration. An explicit
`top_k` can use `strict_top_k=True` for the same boundary-group exclusion rule.

For timing, read `compute.estimate_phase_cycles`,
`schedule.buffer_transition`, `pipeline.estimate_pipeline_cycles`, then
`ranking.apply_ranking_metric`. Ranking recomputes timing with modeled CTA
contention; the pipeline checkpoint also contains a single-CTA timing view.
Missing rates or unsupported schedules remain unknown. Unexpected stage errors
propagate; the autotuner records `analysis_failed` and stops that candidate.

## Imports, profiling, and validation

Package-root exports, settings, signatures, reports, and trace checkpoints are
unchanged. Internal imports now use the source map above; obsolete forwarding
modules are removed. Analysis version 17 and device-profile version 4 remain
unchanged because the equations and schemas are unchanged.

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

## Memory-only ranking

Use `TileTuneConfig(ranking_metric="memory", ...)` to rank by logical memory
work without compute profiles, pipeline timing, or occupancy prediction. The
target's SM count is required. The primary order is (logical byte-waves,
logical access-waves, descending IR pipeline depth). Equal triples share their
group's tail rank; original index only orders report entries. Storage
and dependency facts remain available; unresolved scheduling or a soft register estimate does
not prevent a memory score. Explicit resource policies still apply.

Memory mode uses primitive operator semantics and backend hardware inputs,
without family-policy objects or GEMM/attention recognition. It rejects explicit
GEMM/attention specialization and nonzero attention-specific spill allowances.
Rules for an MMA accumulator or a reduction's source region apply uniformly in
every kernel that contains that operator. The current byte score does not use
the dependency graph or live-storage estimates as a timing prediction.

Analysis version 34 uses the B200 pipeline-depth and strict-selection rules,
with logical request count included in the common primary order. Versioned
`memory.v2` facts retain the requested depth and can be replayed using
`score_memory(accesses, grid_blocks, sm_count, pipeline_depth)`.

This opt-in path scores every oracle winner in the saved 25-case H200 study.
All 25 conservative tail ranks fit a strict 50% budget; the worst is 90/192
(46.875%). The byte-only version reached 20/25. This is a retrospective result
on the frozen FP16/E4M3 kernels and pools, not fresh GPU performance validation.
See [the implementation, results and limitations](../../experiments/MEMORY_RANKING.md).

```python
tuner.set_tiletune_args(True, ranking_metric="memory", alpha=0.5)
```

The common runner accepts `--method top_k --metric memory --alpha 0.5`.

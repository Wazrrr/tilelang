# TileTune code review guide

TileTune reads an actual PrimFunc, propagates its required tiles, estimates
resources and timing, and ranks supplied configurations. Start with
[analysis.py](analysis.py) for the public boundary and [engine.py](engine.py)
for the complete stage order. The shared Python core is in `src/` within this
package; native operator metadata continues to come from the compiler.

## Source map

| Component | Responsibility |
| --- | --- |
| `analysis.py`, `config.py`, `__init__.py` | Public entry points, resolved settings, and exports |
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

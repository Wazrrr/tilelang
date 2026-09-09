# TileTune code review guide

TileTune reads each candidate's actual PrimFunc, recognizes its operation
graph, estimates resources and execution time, and reports a ranking. The source
is organized around **a shared analysis flow and separate kernel-family policies**.

Start with [engine.py](engine.py), then read [families/base.py](families/base.py)
and the family you want to review: [GEMM](families/gemm.py) or
[attention](families/attention.py). For the detailed equations and evidence behind
them, see the [walkthrough](../../docs/tiletune_walkthrough.md).

## Source map

```text
tiletune/
├── __init__.py             Public API
├── config.py               User settings and analysis cache identity
├── analysis.py             Native PrimFunc collection and backward propagation
├── ir_utils.py             Shared read-only queries on captured IR
├── engine.py               Common stage ordering and one register decision
├── trace.py                Optional snapshots at each analysis checkpoint
│
├── families/
│   ├── __init__.py         Recognition order and explicit-family matching
│   ├── base.py             Shared stage implementations and family contract
│   ├── gemm.py             Single-GEMM recognition and consumer restrictions
│   └── attention.py        QK/softmax/PV recognition and attention policies
│
├── register_pressure.py    Initial register-analysis flow and report assembly
├── register_storage.py     Allocation sizes and layout-based storage estimates
├── register_accumulator.py Dense MMA accumulator demand and ownership proof
├── liveness.py             Live tile estimates, including loop-carried state
├── register_policy.py      Physical limits, allowance and final register decision
├── budget.py               Target and user register ceilings
├── warp_specialization.py  Native copy classification and Hopper reservations
│
├── memory.py               External tile traffic and shared allocation sizes
├── shared_storage.py       Shared-buffer lifetimes and storage reuse
├── waves.py                Resource-limited occupancy and CTA waves
├── cost.py                 Combine traffic and occupancy; query device limits
│
├── operation_work.py       Work counts and participants from actual operators
├── reduction.py            Reduction work from fragment ownership
├── service.py              Work / profile rates → operation service cycles
├── pipeline.py             Build pipeline phases and combine their timing
├── tile_schedule.py        Per-buffer readiness/reuse and repeated scheduling
├── cta_work.py             Per-CTA loop counts and whole-grid timing
├── ranking.py              Select metric and sort reported candidates
│
├── profile_schema.py       Accepted performance-model fields and validation
├── device_probes.py        Fixed hardware microbenchmark kernels
├── device_profile.py       Measure, cache and load primitive hardware costs
├── runtime.py              Autotuner session, rejection gates and reports
└── specializations.py      Compatibility imports for the former module path
```

## Shared flow

The entry point is `analysis.analyze_prim_func`. It collects native operations,
propagates output tiles once, establishes the accumulator lower bound from those
same demands, and passes
the collected IR plus CTA tile demands to `engine.run_modules`.

The engine runs the same sequence for each family:

1. Recognize the operation graph through `families.select_specialization`.
2. Add conservative register liveness through the family's `register_pressure`.
3. Predict warp specialization through the family's consumer policy and the
   common native producer classifier.
4. Resolve physical register capacity and logical demand, using the family's
   `register_spill_allowance`. Produce the pre-lowering rejection decision.
5. If ranking is enabled, compute family-selected memory accounting, common
   occupancy, and the family's `pipeline_overlap` analysis.
6. Apply the configured ranking metric and return the module reports.

`runtime.py` enforces rejection and records compiler resource checks. With
`top_k=None`, the autotuner compiles and benchmarks surviving candidates.
With a positive `top_k`, `TileTuneSession.prepare_top_k()` analyzes the full grid,
freezes the first K eligible finite scores, and retains their elaborated PrimFuncs
for compilation. Ties use original index; failures never refill the budget.
All candidates stay in the report, including `not_selected` records.

## What belongs to each family

| Concern | GEMM | Attention | Shared implementation |
| --- | --- | --- | --- |
| Recognition | One dense GEMM | Connected QK → max/exp/sum → PV in one loop | Native collection and dependency facts |
| Operand roles | A, B, accumulator | Q, K, scores, probabilities, V, output accumulator | Buffer identity, never name matching |
| Register estimation | Actual accumulator and other live local tiles | Actual score/output/normalization tiles, including loop state | `register_storage.py`, `register_accumulator.py` and `liveness.py`, coordinated by `register_pressure.py` |
| Soft register allowance | Zero | Configured attention allowance, only after a match | Physical capacity and rejection rules in `register_policy.py` |
| Memory accounting | Backward-propagated input tiles | Each actual external access, avoiding repeated demand paths | Byte counts and storage in `memory.py` |
| Phase labels | GEMM, main loop, outside loop | QK GEMM, softmax/rescale, PV GEMM, outside work | The same operation list feeds liveness and timing |
| WS consumers | Dense GEMMs in a straight-line tile-call loop | Supported internal GEMM/reduce/fill/copy/elementwise work | TMA classification, thread partition and register reservation |
| Pipeline timing | Captured copies and GEMM operations | Captured copies, both GEMMs, and actual softmax/rescaling operations | `operation_work.py` → `service.py` → `pipeline.py` / `tile_schedule.py` |

GEMM and attention currently share the numerical register and timing algorithms.
Their family files document inherited behavior and contain the actual differences.
The base class owns common implementations instead of repeating identical
methods in both files. A future family can override a stage if its model needs
different behavior.

The generic fallback preserves the existing conservative GEMM consumer
restrictions for warp-specialization prediction. This is an execution-policy
fallback; it does not label an unrelated graph as a GEMM or as attention.

## Review register decisions

Read these files in order:

1. [register_pressure.py](register_pressure.py): the short initial analysis flow
   and the existing report fields it assembles.
2. [register_storage.py](register_storage.py): allocation sizes, dtypes, scopes
   and layout-based per-thread estimates. `RegisterStorage.logical_storage`
   contains report entries; `modeled_buffers` maps actual Buffer objects to the
   subset with per-thread allocation estimates.
3. [register_accumulator.py](register_accumulator.py): `_required_accumulators`
   finds full reads with full propagated demands; `_accumulator_ownership`
   establishes the thread bound and replication; `analyze_accumulator_bound`
   converts bits to register units and takes maxima across operations. This
   operator-level proof is shared by GEMM and attention.
4. [liveness.py](liveness.py): after family recognition, `analyze_live_tiles`
   estimates simultaneous storage for explicit/automatic fragments and
   thread-private allocations, including loop-carried state. It produces the
   `tile_liveness` estimate used by demand policy and occupancy. This estimate
   does not strengthen the accumulator proof.
5. The selected family's `register_spill_allowance` and
   [warp_specialization.py](warp_specialization.py): the soft demand margin and
   physical producer/consumer register reservation.
6. [register_policy.py](register_policy.py): the single `analyze_register_policy`
   decision uses physical limits and demand versus consumer capacity plus the
   family allowance. The engine resolves the target/user ceiling before warp
   specialization; the initial pressure stage only describes demand.

The initial `_pressure` call follows this sequence:

```text
analyze_register_storage          → allocation report + modeled Buffer map
analyze_accumulator_bound         → proven per-thread and per-CTA lower bounds
analyze_register_pressure        → assemble the pressure.accumulator checkpoint
```

Buffer names are report labels. Storage scope and native GEMM operand identity
determine which buffers are eligible for the accumulator proof.

Estimated overflow can make a score unknown. Only a proven demand bound or a
resolved physical limit justifies pre-lowering rejection. Compiler register and
spill counters are checked separately in `runtime.check_compiler_resources`.

## Review pipeline timing

Read `pipeline.analyze_pipeline` first to see what information the model receives.
It builds one phase per captured operation, including dependencies and whether
the operation is inside the recognized loop. Family phase names are labels;
they do not insert a fixed GEMM or attention template.

Then follow:

```text
actual operator regions/expressions
    → operation_work.py + reduction.py       work and compute ownership
    → service.py                            operation cycles from profile rates
    → tile_schedule.py                      producer readiness and buffer reuse
    → pipeline.estimate_pipeline_cycles     startup, iterations, outside work
    → cta_work.py + ranking.py               grid timing and final score
```

The serial and overlapping branches use the same operation services. Positive
stage counts require a supported scheduling policy; missing rates or unresolved
IR leave timing unknown. Primitive rates come from `device_profile.py`, whose
fixed probes calibrate hardware costs without replacing the candidate's graph.

## API contracts and validation

Analysis version 15 uses a single tile propagation:

| Entry point/stage | Contract |
| --- | --- |
| `analyze_prim_func(func, config=None, *, ...)` | Analyze every captured global write. No output override; a kernel without captured global output writes raises. |
| `propagate_inputs(func, outputs)` | Query a supplied, nonempty list of Region/BufferRegion roots. Missing or empty roots raise. |
| `_propagate_tiles(col, outputs)` | Receive normalized output tiles, retaining symbolic launch offsets; record demands once for all consumers. |
| `AnalysisContext` | Carry the resolved config, target, device limits, pass settings and trace object. Internal stages require this context. |
| `analyze_register_pressure(col)` | Return allocation, liveness and accumulator demand facts. No preliminary budget decision. |
| `engine.run_modules` | Run the selected family stages; unexpected errors propagate. |
| `TileTuneSession.elaborate` | Record analysis errors as `analysis_failed` and stop the candidate before lowering. |

For migration, remove `outputs` from whole-kernel analysis calls. Use
`propagate_inputs(func, requested_regions)` for a separate region query, and read
`analyze_prim_func(func)["tile_propagation"]` for the kernel's tile demands and per-CTA loop coverage. The old `propagation` field and the
`tile_scope` switch are removed; there is no second traversal or compatibility
report copy. Both public entry points use `_propagate_tiles`.
The `pressure.accumulator` checkpoint now contains demand facts only; budget and
decision fields are available after the engine's register-policy stage.

Pass settings are resolved once by `ir_utils.resolve_pass_configs` and passed
to the family memory, warp-specialization and timing stages. Internal cost and
warp-specialization helpers require the selected family instead of supplying an
alternate default path. The former `effective_pass_configs` helper is removed.

Model validity checks remain: unknown symbolic shapes, unsupported schedules and
unresolved ownership cannot produce a valid numeric estimate. These explicit
unknown results are different from malformed inputs or unexpected stage errors.
The GEMM/attention equations and successful final report fields are unchanged.

Existing tests are in [testing/python/tiletune](../../testing/python/tiletune).
They exercise native propagation, family recognition, liveness, spill allowance,
pipeline scheduling, unknown paths, report ordering, and autotuner integration.

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

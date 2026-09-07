# New Carver: tile analysis and ranking

New Carver analyzes every supplied configuration's actual, elaborated PrimFunc
before lowering. It propagates tile requirements, models register pressure,
shared storage, scheduling, memory traffic and launch waves, then ranks the full
grid. It does not generate configurations, rewrite kernels, or apply a top-K
cutoff. Original Carver and the existing autotune filters remain separate.

## Usage

```python
import tilelang
from tilelang.new_carver import CarverConfig, analyze_prim_func, propagate_inputs

@tilelang.autotune(configs=configs, carver=True)
@tilelang.jit(out_idx=[2])
def kernel(...):
    ...

# Or configure an AutoTuner directly:
tuner.set_carver_args(
    True, mode="report_only", report_path="carver.json",
    ranking_metric="pipeline_time", performance_model=device_profile,
)

# Offline analysis takes an already elaborated PrimFunc; it never runs a kernel.
analysis = analyze_prim_func(func, CarverConfig(), target=target,
                            device_limits=device_limits, pass_configs=pass_configs)
inputs = propagate_inputs(func)
```

Autotuner integration requires CUDA, `tvm_ffi`, and `early_stop=False`. New Carver
cannot run alongside legacy filters or opaque custom compilation hooks. Single
and grouped compilation reuse the analyzed PrimFunc, including normal JIT binding
and per-config pass settings. `report_only` records pressure decisions while
compiling all elaborated candidates; `reject` applies proven pressure failures.
Correctness checks and benchmarking still run normally.

The standalone APIs accept optional output `tirx.BufferRegion` or New Carver
`Region` demands. Propagation keeps offsets and extents, and labels exact regions,
conservative bounds and unknowns. Per-iteration inputs remain separate from the
union across a loop. Queries never shrink the actual allocations or computation.

## Common modules and specializations

```text
Actual PrimFunc → collection and backward input propagation
               → graph-based GEMM / attention / generic specialization
               → register_pressure.py and register_policy.py
               → warp_specialization.py
               → memory.py and shared_storage.py
               → pipeline.py, reduction.py and tile_schedule.py
               → waves.py and ranking.py
```

`engine.py` orchestrates these modules through `specializations.py`. Dense GEMM
has a single matrix operation. Forward attention requires connected QK and PV
operations with intervening max, exponential and sum operations. Recognition
uses the operation graph, not buffer names or configuration names. Set
`specialization="gemm"` or `"attention"` to require a match; the default is `"auto"`.
Unsupported indexing, aliases, calls and scheduling remain unknown and eligible.

The small native adapter exposes existing `ParseOperator`, `GetAccessRegions`
and CUDA producer-copy classification. Python reads reflected operation metadata
and uses TVM arithmetic analysis. The adapter runs no lowering pass.

## Register pressure

Reports separate logical tile storage, modeled demand, physical policy
reservations and actual compiler counters. For a fully demanded dense accumulator
with automatic layout, a lower bound on the maximum per-thread register demand is:

```text
ceil(accumulator_elements × dtype_bits × dtype_lanes / (32 × original_launch_threads))
```

This uses maximally packed 32-bit registers and all original threads as potential
owners. Reduced ownership or replication can increase demand. Explicit layouts
can establish stronger ownership bounds. The accumulator dtype, independently of
input dtype, sets its storage. Pipeline stages and K iterations do not multiply
the accumulator. Shared A/B tiles do not count as equivalent register allocations;
compiler operand fragments and scratch require separate modeling.

Intermediates and repeated accumulator state receive conservative liveness
estimates. Only a proven lower bound above a justified hardware or user budget
can cause pre-lowering rejection. A modeled tile-state estimate is never an upper
bound on all compiler-generated registers.

For supported Hopper pure-TMA pipelines, the compiler policy adds 128 producer
threads to the original consumers. Its register requests are 24 per producer and
240 per consumer for 128 or 256 consumers, or 160 for 384 consumers:

```text
physical_reservation = producer_threads × producer_request
                     + consumer_threads × consumer_request
```

These reservations remain separate from dtype-aware live demand. Manual layouts,
mixed producers and unresolved structure can make the policy unknown. A100 does
not receive Hopper's added threads or its register-reservation policy.

The attention experiment allows 32 additional modeled registers per consumer
(`attention_spill_budget_registers_per_thread=32`). Physical reservations still
obey the SM limit. Demand above the soft allowance makes the ranking uncertain;
only a proven lower bound can reject. This allowance does not quantify PTXAS
spill bytes, and timing currently omits spill traffic. GEMM uses zero allowance.

After compilation, the independent pressure checker records PTXAS registers,
spill loads/stores and stack/local memory. Defaults are zero allowed spill and
local bytes, with no register-count cap. Set `max_spill_bytes=None` and
`max_local_bytes=None` to record without rejecting, as the attention experiments
do. Missing metadata stays unknown. Resource checks do not establish correctness.

## Traffic, waves and pipeline ranking

Propagation follows each demanded compute tile back through copies to global
inputs. For a conventional GEMM CTA, logical traffic is:

```text
BM × K × sizeof(A) + K × BN × sizeof(B) + BM × BN × sizeof(C)
```

Attention charges Q once if loaded outside the loop and K/V per actual tile copy
and loop visit. Internal score/probability tiles do not become global traffic.
Causal loop extents come from the actual IR, including unequal query/key lengths
and partial boundary tiles.

Shared allocation estimates include pipeline copies for buffers written inside
the loop. `shared_storage.py` predicts reuse only for disjoint tile lifetimes;
repeated-loop lifetimes overlap conservatively. Unknown aliases retain the
allocation sum, and `tl.disable_shared_memory_reuse` disables reuse prediction.
Reports retain both the sum and arena estimate. Padding and compiler barriers
remain unmodeled. These estimates cannot cause pre-lowering rejection.

Residency is the minimum of estimated shared-memory, register, thread and block
limits. Hopper policy uses its physical reservation; other paths use tile demand
as an occupancy proxy, not proven physical allocation:

```text
num_waves = ceil(grid_CTAs / (SM_count × resident_CTAs_per_SM))
traffic_waves_score = (logical_traffic_bytes_per_CTA + 1) × num_waves
```

The default metric is `traffic_waves`. Select `pipeline_time` with an explicit
device profile for timing-aware ranking. Matrix work uses aggregate tensor and
shared-memory service ceilings, plus an optional WGMMA per-warpgroup ceiling.
Scalar, exponential and reduction components use:

```text
component_cycles = max(work × concurrent_CTAs / per_SM_rate,
                       work / measured_single_CTA_rate[consumer_threads])
```

Reduction work uses explicit ownership or the compiler's read-only MMA/WGMMA
fragment layout helper. Local combines and lane shuffle/combine pairs have
separate measurements. Unresolved or inter-warp collectives remain unknown.

The pipeline recurrence tracks each producer tile's issue, ready and last-use
release events, with a finite stage ring. K and V can release independently.
Repeated squaring evaluates the recurrence without unrolling all loop iterations.
Uniform workloads multiply wave time by waves. Causal workloads use per-CTA loop
counts and an estimated launch-order dispatch schedule. Hardware scheduling,
cache reuse, instruction subdivisions and dependency chains remain approximations.

All eligible numeric scores precede unknowns, followed by pressure rejections.
Original indices break ties; reports expose the entire tie interval. There is no
fallback mixing byte-wave and cycle scores. Compiler counters and candidate
latencies never enter ranking. Unknown report positions are not predictive ranks.

## Reusable device profiling and A100 coverage

```python
from tilelang.new_carver import profile_device, load_device_profile, anchor_latency

# Explicit GPU operation: fixed primitive kernels, no candidate-grid fitting.
profile = profile_device(input_dtype="float16", cache_path="device.json",
                         memory_regime="streaming")
# Offline, with no device query or execution:
profile = load_device_profile("device.json", input_dtype="float16",
                              memory_regime="streaming")
```

Profile version 4 supports A100 (`sm_80`, FP16/BF16 MMA) and Hopper (`sm_90a`,
FP16/BF16/FP8 WGMMA), with FP32 accumulation. Shared common probes measure cached
and streaming traffic, scalar arithmetic, exponentials, local/lane reductions,
barriers and tile-copy latency. A100 uses a synchronous tile-copy probe; Hopper
uses TMA. Separate matrix probes are keyed by instruction and dtype. Single-CTA
consumer probes cover 32, 64, 128, 256 and 512 threads. These are effective rates
including probe overhead, not guaranteed absolute-latency predictions.

The profile fingerprint includes device type, SM count, architecture, CUDA/driver,
compiler, native build and probe source. Profile each device type locally; do not
reuse H200 rates on A100. Old versions 2 and 3 remain loadable for offline
analysis; `profile_device` requires a matching fingerprint or explicit refresh.
A single optional `anchor_latency(profile, analysis, measured_ms)` changes only a
positive global scale and cannot change configuration ordering.

**Current A100 limit:** stage-zero GEMM and supported attention reductions can
receive MMA timing estimates. Positive-stage Ampere software-pipeline overlap is
not modeled, so its `pipeline_time` scores stay unknown. Pressure, traffic/waves,
compilation, correctness and measurements still run on the entire supplied grid.
A100 FP8 cases are recorded as unsupported because it has no FP8 tensor-core
instructions. An instruction/profile mismatch also leaves a score unknown.

See the [portable experiment instructions](../benchmark/autotune/README.md) for
full 4096³ GEMM, causal/noncausal attention, and varied workloads, with no cutoff.

## Reporting, timing and validation

Each config record includes propagation, pressure evidence, pre-lowering and
post-compile decisions, compiler resources and eventual benchmark status.
`tuner.carver_report` exposes the completed report. An explicit `report_path`
bypasses winner-only cache shortcuts. Settings, device limits, profile and
analysis version enter cache identity; report destinations do not.

Set `TILELANG_AUTOTUNE_TIMING_LOG=/path/timings.tsv` for append-only orchestration
records covering elaboration, analysis, lowering, code generation, NVCC, adapter
creation, queue waits and profiler work. Reports separate summed stage-cost
percentages from end-to-end wall time: parallel or nested stage durations must
not be added and interpreted as wall time.

Build the native adapter before use with the repository development workflow:

```bash
cmake -S . -B build
cmake --build build -j8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pytest testing/python/new_carver/ -q
```

Analysis version 13 adds A100 profile selection and MMA reduction-map prediction
to the version-12 model. The [v12 H200 findings](new_carver_validation.md) summarize
the preceding experiments and known limitations. A100 runtime/performance results
must be collected on an A100 node; cross-compilation alone is not that validation.

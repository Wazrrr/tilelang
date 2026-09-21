# H200: exhaustive, multi-GPU, and pipelined TileTune experiments

Status: runner implemented; execution paused at the user's request because no
GPUs are available. No experiment is queued to start automatically. Full-sweep
results and oracle retention are pending.

The grouped-compile recovery below is implemented and covered by focused
failure-isolation tests and H200 pipeline/alpha tests in
`testing/python/autotune/test_grouped_compile_fallback.py`. The sequential study
runner is `python -m experiments.common.h200`; it wires the exact E1/E2/E3 modes
and runs monitored preflights before the complete oracle-retention study.

Use `dev-h200-new`, including grouped-compile recovery and the KDA intra-chunk
migration. Freeze the actual code revision after the runner changes below. Use the current
[benchmark contract](BENCHMARK_CONTRACT.md), contract version 3 and configuration
space version 8. The older FP16/E4M3 29,200-candidate study does not establish
oracle retention for this BF16/block-scaled suite.

## Experiment matrix

Run all 25 final workloads once per experiment: 75 workload runs altogether.

| Experiment | Selection | Compiler workers | Benchmark GPUs | Pipeline | Grouped compilation | Post-compile TileTune policy |
| --- | --- | ---: | ---: | --- | --- | --- |
| E1: exhaustive, one GPU | Complete pool | 128 | 1 | Off | Off | Off |
| E2: exhaustive, four GPUs | Complete pool | 128 | 4 | Off | Off | Off |
| E3: TileTune, four GPUs | Unified memory score, strict alpha=0.5 | 128 | 4 | On | On, size 8 | On |

Multi-GPU means distributing candidate benchmarks within each workload through
`AutoTuner.run(benchmark_multi_gpu=True, benchmark_devices=[0,1,2,3])`.
It does not mean running four separate workloads concurrently. E1 uses
`benchmark_multi_gpu=False`. Pipeline means overlapping compilation and
benchmarking with `use_pipeline=True`; it is separate from the kernel's tuned
`num_stages` and from the `pipeline_time` ranking metric. E3 uses
`ranking_metric="memory"`, not `pipeline_time`.

Keep `enable_grouped_compile=False` in E1/E2. In E3, set
`enable_grouped_compile=True, group_compile_size=8`; groups contain at most eight
selected configs with compatible effective compiler settings. Keep
`early_stop=False` in all three runs.
E1 and E2 do not apply spill-based pruning, so their oracle tables include all
successfully compiled and numerically correct candidates, including spillers.
E3 retains the existing post-compile checks: spill/local byte allowances of 0
for GEMM, grouped GEMM and KDA, 64 for attention, and 128 for FP8 GEMM. Hardware
limits also apply. Use `common/resource_policy.py`; do not enable legacy filters.

## Workload and budget inventory

Use `common.spec.default_workloads(smoke=False)` and each family's complete
`spaces.get_configs()` pool. Preserve config ordering and SHA-256 config IDs.

| Family | Five final workload names | Pool per workload | E3 maximum selected per workload |
| --- | --- | ---: | ---: |
| GEMM | gemm_decode, gemm_prefill, gemm_ffn_down, gemm_square, gemm_square_large | 3456 | 1728 |
| Attention | attention_short_causal, attention_batched_causal, attention_noncausal, attention_causal, attention_long_causal | 576 | 288 |
| KDA intra-chunk | kda_intra_short, kda_intra_medium, kda_intra_regular, kda_intra_batched, kda_intra_long | 512 | 256 |
| FP8 GEMM | gemm_fp8_decode, gemm_fp8_prefill, gemm_fp8_ffn_down, gemm_fp8_square, gemm_fp8_square_large | 576 | 288 |
| Grouped GEMM | grouped_gemm_decode, grouped_gemm_prefill, grouped_gemm_aligned, grouped_gemm_down_aligned, grouped_gemm_ragged | 576 | 288 |

Each exhaustive experiment attempts 28,480 candidates. E3 elaborates/analyzes
all 28,480 and selects at most 14,240 for compilation. Whole equal-score groups
must fit inside `floor(0.5 * original_pool_size)`. A group crossing the boundary
is excluded; ties are not split or expanded. Failures and unknown scores stay
in the original denominator. Do not refill after analysis, compilation,
post-compile rejection, correctness, or benchmark failures.

There are at most 71,200 candidate slots submitted for compilation across
the three full experiments, excluding preflight and winner verification.
Failed shared builds add retry attempts, which must be counted and timed.
Actual benchmark counts will be smaller when candidates fail or are filtered.

KDA inputs are BF16 Q/K of shape (B,S,H,128), FP32 gates of the same shape,
and BF16 beta of shape (B,S,H). Outputs are BF16 Aqk (B,S,H,64) and
Akk (B,S,H,16). Head dimension=128, chunk=64, sub-chunk=16.

| KDA workload | B | S | H |
| --- | ---: | ---: | ---: |
| kda_intra_short | 1 | 2048 | 32 |
| kda_intra_medium | 1 | 4096 | 64 |
| kda_intra_regular | 1 | 8192 | 32 |
| kda_intra_batched | 2 | 4096 | 32 |
| kda_intra_long | 1 | 16384 | 64 |

Use block_H=1–16, stages=0–7, threads={32,64,128,256}: 512 configs per
workload. This includes all 32 original example configs (block_H={1,2,4,8},
stages=0–3, threads={128,256}). See [the KDA contract](kda/README.md).
Previous chunk-output oracle and spill validations do not establish retention
for this intra-chunk operation.

E3's grouped compiler isolates per-config elaboration/lowering failures and
post-compile rejections. If a shared device/host build fails, it bisects the
unfinished configs and retries down to singleton builds, reusing their lowered
IR and effective settings. Already rejected or completed configs stay final.
Only a failing singleton is marked as a compilation failure; valid neighbors
remain available for benchmarking. Retries stay inside the original selection,
never refill the alpha budget, and include failed-attempt time in compile costs.

## Shared measurement settings

- Use four idle H200 GPUs of the same model for E2/E3 and the first of those
  devices for E1. Bind explicit physical GPU UUIDs and record the logical mapping.
  Process workloads sequentially and experiments in E1, E2, E3 order.
  Use GPUs 0–3 for this study, leaving GPUs 4–7 unused by the study. Never launch
  a second workload, independent experiment, training job, or preflight alongside
  the current one. A host lease and leases on all four GPUs reject a second
  cooperating launcher, even one requesting a disjoint device set.
- Set `TILELANG_AUTO_TUNING_CPU_COUNTS=128` and
  `TILELANG_AUTO_TUNING_MAX_CPU_COUNT=128`. Check that the tuner's resolved worker
  count is exactly 128; CPU affinity or allocation can otherwise silently clamp
  the requested count. This is one shared 128-worker compiler pool per workload,
  not 128 workers per GPU.
- Use fresh worker processes with `TILELANG_DISABLE_CACHE=1` and
  `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`. Give each run a new output directory.
  Keep compiler options, authoritative example builders, inputs, references,
  tolerances and benchmark backend identical across the three experiments.
- Fix input seed 123, event timing, `warmup=10` ms, `rep=50` ms,
  and a 60-second per-candidate benchmark timeout. These are the profiler's time
  budgets; it determines iteration counts automatically. Preserve the same input and
  cache-conditioning behavior in all modes. Record those settings explicitly.
- Keep memory diagnostics opt-in/off. E3 receives the same generic integer
  `input_values` metadata used by the existing runner, and queried H200 device
  limits. Supply the full pool and `alpha=0.5`; do not substitute a top-K subset.
- Monitor GPU/process activity through `experiments.utils.monitor.run_monitored`.
  Discard and retry contaminated workload runs, preserving their logs. Record
  device UUIDs, clocks, driver/toolchain versions, source hashes and pool hashes.
  Pin workers to one frozen set of sibling-complete CPU cores with at least 136
  logical CPUs: 128 compiler workers plus eight CPUs of benchmark/runtime
  headroom. Place the coordinator outside that set and set OpenMP, MKL,
  OpenBLAS, NumExpr, TVM and Torch thread counts to one to prevent nested thread
  pools. CPU affinity is not an exclusive OS reservation: sample CPU busy time
  and subtract the worker process group's own CPU usage. Require five quiet
  one-second samples before starting. Reject two consecutive samples exceeding
  two external busy CPU cores, one I/O-wait core, or 0.1 stolen CPU cores. These
  small allowances cover scheduler/monitor noise and are recorded in the audit.
  Foreign GPU activity and monitor gaps invalidate the attempt immediately.
  Never kill or reconfigure unrelated jobs. Contended attempts are retried at
  most three times per invocation, waiting for quiet resources before each retry.

## Sequential execution and interruption recovery

```bash
# Planning imports no CUDA/compiler packages and creates no results.
python -m experiments.common.h200 --plan

# Runs 15 small preflights, then the 75 full workload runs, one at a time.
python -m experiments.common.h200 --gpus 0 1 2 3 \
  --output experiments/results/h200-three-run-20260921

# Same command and output, with --resume, after Ctrl-C/SIGTERM or a restart.
python -m experiments.common.h200 --gpus 0 1 2 3 \
  --output experiments/results/h200-three-run-20260921 --resume
```

Use the `tl` environment. The runner selects/binds the GPUs; a wrapper must not
reduce CUDA visibility to one GPU. It runs preflight and full-sweep phases
sequentially, with no background compilation from another workload. Preflight
tries at most eight candidates per family/mode. E3 still analyzes/selects from
the complete pool, then marks selected candidates beyond those eight as
`preflight_omitted`; preflight results never establish an exhaustive oracle.
Workers explicitly import TileLang from this checkout, overriding any editable
installation pointing to another branch only within that worker process.

The frozen manifest includes source/native-build identity, settings, complete
ordered pools, interpreter, CPU affinity and GPU UUIDs. Resume rejects changed
identities. Each workload has immutable numbered attempt directories and an
atomic completion marker containing result hashes. Only a complete terminal
outcome set with an uncontended monitor completion can be reused. SIGINT,
SIGTERM and SIGHUP stop the current owned worker process group and preserve
all earlier completed workloads. Worker parent-death cleanup and inherited
lease descriptors prevent an orphaned worker from overlapping a new launcher.

An interrupted or contaminated workload restarts from the beginning with cold
caches. Partial compilation/benchmark logs remain available for diagnosis but
are not spliced into a measured end-to-end run. This preserves valid timing
comparisons. Atomic progress files report the phase, active workload/attempt and
completed count. Stale partial attempts are ignored on resume; completed files
are checked for identity, content hashes, counts, outcomes and selection budget.
Queue/idle waits and discarded attempts remain separate from accepted tuning
times. The final queue step writes the E1/E2 oracle-retention CSV and JSON.

## Runner preparation and execution order

1. Use `experiments/common/h200.py` as the coordinator and
   `experiments/common/system.py` as its fresh-process worker. The worker's
   `baseline`, `multi_gpu` and `tiletune` modes implement E1/E2/E3. The older
   `combined` ablation still does not imply TileTune; do not substitute it for E3.
2. Make the runner save complete per-config outcomes and TileTune reports,
   including selection IDs, scores, equal-score tail ranks, effective resource
   policy, compiler resource counters, rejection reasons, correctness and
   benchmark status. Record and assert the effective worker count and GPU count.
   Keep `--plan` standard-library-only, without GPU queries or result creation.
3. Validate the runner with focused offline checks for mode settings, full-pool
   identity, strict alpha selection and oracle comparison. Run a monitored GPU
   preflight for each family and mode using known valid representative configs;
   test E3 selection against its full original pool. Store preflight separately.
4. Freeze a manifest containing all 25 workload definitions, full ordered config
   pools/IDs, code identity, settings and resolved devices. Execute the three
   complete experiments using that manifest. Preserve every failure as an
   outcome; an interrupted or incomplete sweep cannot establish an oracle.
5. Run the retention audit below, then remeasure each distinct E1/E2/E3 winner
   seven times on the same single H200 with matching inputs/timing settings.
   Report median and spread separately from the original sweep measurements.

## Oracle-retention audit

For each workload, derive the minimum valid measured latency from each complete
exhaustive run. Retain every config attaining that minimum, and take the union
of E1 and E2 oracle config IDs. This tests both measured oracles when timing or
GPU differences produce different winners. Do not redefine an oracle after
applying E3's post-compile filter or after winner remeasurement.

For every oracle config in that union, check:

1. E3 analyzed the identical configuration in an identical workload/pool, with a
   finite eligible memory score and equal-score tail rank no greater than
   `floor(pool_size / 2)`.
2. Its exact config ID is in E3's saved selected set.
3. Its E3 compilation and post-compile checks passed, with compiler-resource
   evidence sufficient to verify the configured limits.
4. Its E3 numerical correctness check and benchmark completed successfully.

Produce `oracle_retention.csv` and JSON with workload, baseline experiment,
oracle config ID/config, oracle latency, pool size, alpha budget, memory score,
tail rank/fraction, selection membership, compiler counters, post-compile
decision, correctness, benchmark status and failure reason.

Success requires **25/25 workloads retaining every E1/E2 oracle config through
all four checks**, with no selected set exceeding half its original pool.
Report selection retention separately from post-compile and final usable
retention. If a check fails, keep the original result and diagnose the failing
stage; do not silently change alpha, score, spill allowances, or replace configs.
The spill allowances were validated on the older suite and must be rechecked
against these newly measured oracles.

## Timing report and artifacts

Write outputs under a new `experiments/results/h200-three-run-<timestamp>/`
directory, separated by experiment and workload. Save the frozen manifest,
compiler/benchmark logs, all config outcomes, TileTune reports, monitor evidence,
retention CSV/JSON, winner remeasurements and a final comparison table.

Report per-workload and total tuning wall time, E1/E2 and E1/E3 speedups, full
worker wall time, elaboration/analysis time, compilation time, benchmark time,
selected/rejected/success/failure counts and best latency. Include E3's complete
analysis and selection cost in tuning time. Separate overlapping stage timings
from elapsed wall time; do not sum parallel work as though it were serial.
Keep verification and preflight costs separate from tuning costs.

For any accompanying XGBoost comparison, include training in the primary
end-to-end timing: training/validation sample collection (including compilation,
correctness checks and benchmarking), feature preparation, model fitting with
validation/early stopping, model loading, prediction/selection, and selected
candidate compilation/checking/benchmarking. Measure elapsed wall time across
these stages; do not substitute prediction-only timing or sum overlapping work.
Report collection, fitting, and online tuning times separately as a breakdown.
Keep held-out oracle collection and post-run winner verification separate from
the method's end-to-end time, and never use held-out labels for training.

Charge each shared training/validation collection and model-training run once
in the suite total. Report its reuse scope and the online cost for each workload;
any amortized per-workload figure must show the allocation explicitly. Reusing
a frozen model must carry its recorded collection/training cost into a reported
training-inclusive comparison, rather than treating training as free. If that
cost is unavailable, mark training-inclusive timing incomplete. E1/E2/E3 remain
the three modes above; this accounting rule applies when reporting an XGBoost
baseline alongside them.

Report shortlist quality against each exhaustive table using that table's own
latencies, and report common-GPU winner remeasurements separately. E2/E3 compares
the combined effect of pipeline, grouped compilation and TileTune; these three
experiments alone cannot attribute that speedup independently to each feature.

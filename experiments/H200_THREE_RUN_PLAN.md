# H200: exhaustive, multi-GPU, and pipelined TileTune experiments

Status: runner implemented for CUPTI timing; execution remains a manual action
after the four-GPU preflight passes. No experiment is queued to start
automatically. Full-sweep results and oracle retention are pending.

The grouped-compile recovery below is implemented and covered by focused
failure-isolation tests and H200 pipeline/alpha tests in
`testing/python/autotune/test_grouped_compile_fallback.py`. The resource-scheduled study
runner is `python -m experiments.common.h200`; it wires the exact E1/E2/E3 modes
and runs monitored preflights before the complete oracle-retention study.

Use `dev-h200-new`, including grouped-compile recovery and the KDA intra-chunk
migration. Freeze the actual code revision after the runner changes below. Use the current
[benchmark contract](BENCHMARK_CONTRACT.md), contract version 4 and configuration
space version 11. The older FP16/E4M3 29,200-candidate study does not establish
oracle retention for this BF16/original-FP8 suite.

## Experiment matrix

Run all 25 final workloads once per experiment: 75 workload runs altogether.

| Experiment | Selection | Compiler workers/workload | Benchmark GPUs/workload | Concurrent workloads | Timing | Pipeline | Grouped compilation | Post-compile policy |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| E1: exhaustive, one GPU | Complete pool | 64 | 1 | 1 | CUPTI | Off | Off | No pruning; defines the one-GPU oracle |
| E2: exhaustive, four GPUs | Complete pool | 64 | 4 | 1 | CUPTI | Off | Off | No pruning; defines the four-GPU oracle |
| E3: TileTune, four GPUs | Unified memory score, strict alpha=0.5 | 64 | 4 | 1 | CUPTI | On | On, size 8 | Enforced after compilation |

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
E1 and E2 do not apply spill-based pruning, so their oracle tables remain
independent of the policy being audited and include every successfully compiled,
numerically correct candidate. E3 turns the post-compile resource check on:
spill/local byte allowances are 0 for GEMM, grouped GEMM and KDA, 64 for
attention, and 128 for FP8 GEMM. Hardware register and launch limits also apply.
The E3 report must contain PTXAS counters and a post-compile decision for every
candidate that reaches that stage. Use `common/resource_policy.py`; do not enable
the legacy kernel-family filters. If an E1/E2 oracle fails this E3 policy, the
retention result fails rather than redefining the oracle.

## Workload and budget inventory

Use `common.spec.default_workloads(smoke=False)` and each family's complete
`spaces.get_configs()` pool. Preserve config ordering and SHA-256 config IDs.
The tables below come from each family's `cases.py`. Every active pool is the
recorded SM90a compilation intersection for all five final workloads in
[experiments/compilation](compilation); this establishes compilation validity,
while the three runs still perform independent correctness and timing checks.

| Family | Five final workload names | Pool per workload | E3 maximum selected per workload |
| --- | --- | ---: | ---: |
| GEMM | gemm_decode, gemm_prefill, gemm_ffn_down, gemm_square, gemm_square_large | 576 | 288 |
| Attention | attention_short_causal, attention_batched_causal, attention_noncausal, attention_causal, attention_long_causal | 512 | 256 |
| KDA intra-chunk | kda_intra_short, kda_intra_medium, kda_intra_regular, kda_intra_batched, kda_intra_long | 645 | 322 |
| FP8 GEMM | gemm_fp8_decode, gemm_fp8_prefill, gemm_fp8_ffn_down, gemm_fp8_square, gemm_fp8_square_large | 576 | 288 |
| Grouped GEMM | grouped_gemm_decode, grouped_gemm_prefill, grouped_gemm_aligned, grouped_gemm_down_aligned, grouped_gemm_ragged | 576 | 288 |

Each exhaustive experiment attempts 14,425 candidates. E3 elaborates/analyzes
all 14,425 and selects at most 7,210 for compilation. Whole equal-score groups
must fit inside `floor(0.5 * original_pool_size)`. A group crossing the boundary
is excluded; ties are not split or expanded. Failures and unknown scores stay
in the original denominator. Do not refill after analysis, compilation,
post-compile rejection, correctness, or benchmark failures.

There are at most 36,060 candidate slots submitted for compilation across
the three full experiments, excluding preflight and winner verification.
Failed shared builds add retry attempts, which must be counted and timed.
Actual benchmark counts will be smaller when candidates fail or are filtered.

### 1. BF16 GEMM

The experiment adapter is [gemm/kernel.py](gemm/kernel.py), and the TileLang
kernel is `make_autotune_kernel_builder` in
[examples/gemm/example_gemm_advanced_autotune.py](../examples/gemm/example_gemm_advanced_autotune.py).
It computes `C=A@B.T`, where A is BF16 `[M,K]`, B is BF16 `[N,K]`, C is BF16
`[M,N]`, and accumulation is FP32. B is already transposed in storage; input
construction and reference computation are outside the benchmark interval.

| Workload | M | N | K | A | B | C |
| --- | ---: | ---: | ---: | --- | --- | --- |
| gemm_decode | 256 | 4096 | 4096 | `[256,4096]` | `[4096,4096]` | `[256,4096]` |
| gemm_prefill | 1024 | 4096 | 4096 | `[1024,4096]` | `[4096,4096]` | `[1024,4096]` |
| gemm_ffn_down | 1024 | 4096 | 14336 | `[1024,14336]` | `[4096,14336]` | `[1024,4096]` |
| gemm_square | 4096 | 4096 | 4096 | `[4096,4096]` | `[4096,4096]` | `[4096,4096]` |
| gemm_square_large | 4096 | 14336 | 4096 | `[4096,4096]` | `[14336,4096]` | `[4096,14336]` |

The 576-config pool comes from [gemm/spaces.py](gemm/spaces.py). It uses
`block_M={64,128,256}`, `block_N={32,64,96,128,192,256}`,
`block_K={32,64}`, `num_stages=0..3`, `thread_num={128,256}`, and
rasterization on/off. This compiler-qualified H200 pool contains the complete
original 288-config example pool unchanged and adds the N tiles 32, 96 and 192.

### 2. FP8 GEMM

The adapter is [gemm_fp8/kernel.py](gemm_fp8/kernel.py), and the kernel is
`matmul` in
[examples/gemm_fp8/example_tilelang_gemm_fp8.py](../examples/gemm_fp8/example_tilelang_gemm_fp8.py).
It computes `C=A@B.T` with E4M3FN A `[M,K]`, E4M3FN B `[N,K]`, E4M3FN C
`[M,N]`, and FP32 accumulation. There are no scale tensors. The five workloads
use the same `(M,N,K)` values and tensor shapes as BF16 GEMM above.

The 576-config pool in [gemm_fp8/spaces.py](gemm_fp8/spaces.py) is the product
of `block_M={64,128,256}`, `block_N={64,128,256}`, `block_K={32,64}`,
`num_stages=0..7`, `threads={128,256}`, and rasterization on/off. It contains
the complete original pool from
[examples/gemm_fp8/example_gemm_fp8_tiletune.py](../examples/gemm_fp8/example_gemm_fp8_tiletune.py).

### 3. Grouped BF16 GEMM

The adapter is [grouped_gemm/kernel.py](grouped_gemm/kernel.py), and the kernel
is `grouped_gemm` in
[examples/grouped_gemm/example_grouped_gemm_fwd.py](../examples/grouped_gemm/example_grouped_gemm_fwd.py).
For group sizes `Mi`, A is concatenated BF16 `[sum(Mi),K]`. With
`transpose_b=false`, B is BF16 `[G,K,N]`; with `transpose_b=true`, it is
`[G,N,K]`. The result is BF16 `[sum(Mi),N]` with FP32 accumulation. Batch
sizes, offsets, and padded offsets are prepared outside the timed kernel.

| Workload | Group sizes `Mi` | G | sum(Mi) | N | K | transpose B | A shape | B shape | C shape |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| grouped_gemm_decode | `[1,2,4,8]` | 4 | 15 | 2048 | 7168 | false | `[15,7168]` | `[4,7168,2048]` | `[15,2048]` |
| grouped_gemm_prefill | `[32,32,32,32,32,32,32,32]` | 8 | 256 | 2048 | 7168 | false | `[256,7168]` | `[8,7168,2048]` | `[256,2048]` |
| grouped_gemm_aligned | `[128,128,128,128]` | 4 | 512 | 2048 | 7168 | false | `[512,7168]` | `[4,7168,2048]` | `[512,2048]` |
| grouped_gemm_down_aligned | `[256,256,256]` | 3 | 768 | 7168 | 2048 | true | `[768,2048]` | `[3,7168,2048]` | `[768,7168]` |
| grouped_gemm_ragged | `[63,77,111,280]` | 4 | 531 | 7168 | 2048 | true | `[531,2048]` | `[4,7168,2048]` | `[531,7168]` |

The 576-config pool in [grouped_gemm/spaces.py](grouped_gemm/spaces.py) uses
fixed `block_M=64`, `block_N={32,64,96,128,192,256}`,
`block_K={16,32,48,64,96,128}`, `num_stages=0..7`, and
`threads={128,256}`.

### 4. BF16 FlashAttention

The adapter is [flash_attention/kernel.py](flash_attention/kernel.py), and the
kernel is `flashattn` in
[examples/flash_attention/example_mha_fwd_bshd.py](../examples/flash_attention/example_mha_fwd_bshd.py).
Q, K, V, and O are BF16 BSHD tensors `[B,S,H,D]`. The kernel uses FP32 online
softmax/accumulation and the workload's causal flag.

| Workload | B | S | H | D | Causal | Q/K/V/O shape |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| attention_short_causal | 1 | 512 | 32 | 64 | true | `[1,512,32,64]` |
| attention_batched_causal | 2 | 2048 | 16 | 64 | true | `[2,2048,16,64]` |
| attention_noncausal | 1 | 4096 | 32 | 128 | false | `[1,4096,32,128]` |
| attention_causal | 1 | 4096 | 32 | 128 | true | `[1,4096,32,128]` |
| attention_long_causal | 1 | 8192 | 16 | 128 | true | `[1,8192,16,128]` |

The candidate grid in [flash_attention/spaces.py](flash_attention/spaces.py)
has 1,024 entries: `block_M={32,64,128,256}`, `block_N=16..256` in steps of
16, `num_stages=0..7`, and `threads={128,256}`. The common H200 compilation
intersection is the active 512-config pool: `block_M=64,threads=128`;
`block_M=128,threads={128,256}`; and `block_M=256,threads=128`, with every N
tile and stage for each retained pair.

### 5. BF16 KDA intra-chunk

The adapter is [kda/kernel.py](kda/kernel.py), and the kernel is
`tilelang_chunk_kda_fwd_intra_token_parallel` in
[examples/kda/chunk_intra_token_parallel.py](../examples/kda/chunk_intra_token_parallel.py).
Inputs are BF16 Q/K `[B,S,H,128]`, FP32 cumulative gates of the same shape, and
BF16 beta `[B,S,H]`. Outputs are BF16 Aqk `[B,S,H,64]` and Akk
`[B,S,H,16]`; accumulation is FP32. Head dimension is 128, chunk size is 64,
sub-chunk size is 16, and scale is `128**-0.5`. Gate construction is outside
the benchmark interval.

| Workload | B | S | H | Q/K/gate shape | beta shape | Aqk shape | Akk shape |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| kda_intra_short | 1 | 2048 | 32 | `[1,2048,32,128]` | `[1,2048,32]` | `[1,2048,32,64]` | `[1,2048,32,16]` |
| kda_intra_medium | 1 | 4096 | 64 | `[1,4096,64,128]` | `[1,4096,64]` | `[1,4096,64,64]` | `[1,4096,64,16]` |
| kda_intra_regular | 1 | 8192 | 32 | `[1,8192,32,128]` | `[1,8192,32]` | `[1,8192,32,64]` | `[1,8192,32,16]` |
| kda_intra_batched | 2 | 4096 | 32 | `[2,4096,32,128]` | `[2,4096,32]` | `[2,4096,32,64]` | `[2,4096,32,16]` |
| kda_intra_long | 1 | 16384 | 64 | `[1,16384,64,128]` | `[1,16384,64]` | `[1,16384,64,64]` | `[1,16384,64,16]` |

The raw grid in [kda/spaces.py](kda/spaces.py) has 1,024 entries from
`block_H=1..16`, `num_stages=0..15`, and `threads={32,64,128,256}`. Use its
645-config common H200 compilation intersection. It includes all 32 original
example configs (`block_H={1,2,4,8}`, `num_stages=0..3`,
`threads={128,256}`). Previous oracle and spill validations do not establish
retention for this intra-chunk operation.

E3's grouped compiler isolates per-config elaboration/lowering failures and
post-compile rejections. If a shared device/host build fails, it bisects the
unfinished configs and retries down to singleton builds, reusing their lowered
IR and effective settings. Already rejected or completed configs stay final.
Only a failing singleton is marked as a compilation failure; valid neighbors
remain available for benchmarking. Retries stay inside the original selection,
never refill the alpha budget, and include failed-attempt time in compile costs.

## Shared measurement settings

- Run experiment phases in E1, E2, E3 order. The initial E1 process uses GPU 0,
  one 72-CPU workload pool (64 compiler workers plus eight CPUs of runtime
  headroom), and separate CPUs for the coordinator. It processes the 25 E1
  workloads sequentially. E2/E3 use all four benchmark GPUs.
  Bind explicit physical GPU UUIDs and record the logical mapping. A host lease
  and leases on the allocated GPUs reject a second cooperating experiment
  launcher. Monitor each workload's active GPU subset.
  Wait before launch and discard that attempt if a foreign process or GPU activity
  appears on any GPU assigned to it.
- Set `TILELANG_AUTO_TUNING_CPU_COUNTS=64` and
  `TILELANG_AUTO_TUNING_MAX_CPU_COUNT=64`. Check that the tuner's resolved worker
  count is exactly 64; CPU affinity or allocation can otherwise silently clamp
  the requested count. This is one shared 64-worker compiler pool per workload,
  not 64 workers per GPU.
- Use fresh worker processes with `TILELANG_DISABLE_CACHE=1` and
  `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`. Give each run a new output directory.
  Keep compiler options, authoritative example builders, inputs, references,
  tolerances and benchmark backend identical across the three experiments.
- Fix input seed 123, CUPTI timing, `warmup=10` ms, `rep=50` ms, and a
  60-second per-candidate benchmark timeout. These are profiler time budgets;
  it determines iteration counts automatically. The reported latency must come
  from CUPTI CUDA activity, divided by the repeat count after excluding only the
  annotated 256 MiB L2-cache flush ranges. The profiler may use a five-iteration
  CUDA-event estimate solely to choose warmup/repeat counts; that estimate is
  never the reported candidate latency. Preserve the same input and cache
  conditioning in all modes, and record `backend="cupti"` in every result.
- Keep memory diagnostics opt-in/off. E3 receives the same generic integer
  `input_values` metadata used by the existing runner, and queried H200 device
  limits. Supply the full pool and `alpha=0.5`; do not substitute a top-K subset.
- Monitor GPU/process activity through `experiments.utils.monitor.run_monitored`.
  Discard and retry contaminated workload runs, preserving their logs. Record
  device UUIDs, clocks, driver/toolchain versions, source hashes and pool hashes.
  Pin each workload to one frozen sibling-complete set of at least 72 logical
  CPUs: 64 compiler workers plus eight CPUs of benchmark/runtime headroom. Place
  the coordinator outside that workload pool. Set OpenMP, MKL,
  OpenBLAS, NumExpr, TVM and Torch thread counts to one to prevent nested thread
  pools. CPU affinity is not an exclusive OS reservation: sample CPU busy time
  and subtract the worker process group's own CPU usage. Record samples above
  two external busy CPU cores, one I/O-wait core, or 0.1 stolen CPU cores, but
  treat CPU contention as advisory because other host work does not use a shared
  resource allocator. It does not delay launch or invalidate measurements.
  Foreign GPU activity and monitor gaps invalidate the attempt immediately.
  Never kill or reconfigure unrelated jobs. GPU-contended attempts are preserved
  and retried until a clean attempt completes or the coordinator is interrupted.
  Owned CPU accounting traverses only each worker's process tree, so its polling
  cost does not grow with unrelated process entries on a shared host.

## Local execution and interruption recovery

```bash
# Planning imports no CUDA/compiler packages and creates no results.
python -m experiments.common.h200 --plan

# Run five E1 preflights followed by all 25 E1 workloads on GPU 0.
experiments/run_h200_e1.sh experiments/results/h200-e1-20260921

# Resubmit the same command and result path after cancellation or a node restart;
# the launcher detects the frozen manifest and enables verified resume.
```

The E1 launcher activates the `tl` environment. The runner selects and binds GPU
0 and runs the E1 preflight and full sweep without workload overlap, one workload
at a time. CPU contention is retained in `monitor.json` as advisory evidence;
foreign GPU use still causes the complete workload attempt to be discarded and
retried. Preflight
tries at most eight candidates per family/mode. E3 still analyzes/selects from
the complete pool, then marks selected candidates beyond those eight as
`preflight_omitted`; preflight results never establish an exhaustive oracle.
Before accepting preflight, verify CUPTI activity is available on all four
devices and every worker artifact records `measurement.backend="cupti"`.
Workers explicitly import TileLang from this checkout, overriding any editable
installation pointing to another branch only within that worker process.

The frozen manifest includes source/native-build identity, settings, complete
ordered pools, interpreter, CPU affinity and GPU UUIDs. Resume rejects changed
identities. Each workload has immutable numbered attempt directories and an
atomic completion marker containing result hashes. Only a complete terminal
outcome set with an uncontended monitor completion can be reused. SIGINT,
SIGTERM and SIGHUP signal every active resource slot and preserve all earlier
completed workloads. Worker parent-death cleanup and inherited lease descriptors
prevent an orphaned worker from overlapping a new launcher.

Each interrupted or contaminated workload restarts from the beginning with cold
caches. Partial compilation/benchmark logs remain available for diagnosis but
are not spliced into a measured end-to-end run. This preserves valid timing
comparisons. Atomic progress files report the phase, all active workloads and
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
   benchmark status. Record and assert the effective worker count, GPU count,
   and CUPTI backend. Keep `--plan` standard-library-only, without GPU queries
   or result creation.
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

For any accompanying XGBoost comparison, freeze two training shapes and one
validation shape per family from each family's `training_cases()`; these 15
shapes are disjoint from the 25 final workloads. Use deterministic seeded config
hash sampling at fraction 0.1 and seed 123. This attempts `ceil(0.1*pool_size)`
configs per shape, and failed samples consume the budget without replacement:

| Family | Per training/validation shape | Two training shapes | One validation shape |
| --- | ---: | ---: | ---: |
| GEMM | 58 | 116 | 58 |
| Attention | 52 | 104 | 52 |
| KDA intra-chunk | 65 | 130 | 65 |
| FP8 GEMM | 58 | 116 | 58 |
| Grouped GEMM | 58 | 116 | 58 |
| Total | 291 | 582 | 291 |

Compile, check, and benchmark these samples with the same CUPTI measurement
contract. Fit the CPU `hist` regressor to log latency with at most 600 rounds,
maximum depth 10, learning rate 0.05, subsample 0.8, seed 123, 64 threads, and
early stopping after 20 validation rounds. Run collection and fitting without
overlapping an experiment workload or another CPU/GPU job.

The primary XGBoost end-to-end wall time is
`T_train_collection + T_validation_collection + T_feature_preparation +`
`T_fit_and_early_stop + T_model_save_load + T_final_feature_preparation +`
`T_predict_and_select + T_selected_compile_check_benchmark`. Start the timer
before the first training sample is prepared and stop it after the final
selected candidate is benchmarked. Record elapsed wall time directly around
each serial phase; do not substitute prediction-only timing or add CPU/GPU stage
durations that overlapped. Report collection, fitting, and online tuning as the
breakdown. Keep held-out exhaustive oracle collection and post-run winner
verification outside this method time, and never use final-workload labels for
training.

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

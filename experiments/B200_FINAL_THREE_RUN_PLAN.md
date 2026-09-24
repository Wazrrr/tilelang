# B200 final three-experiment plan

The execution and acceptance contracts are split into three standalone plans:

- [E1: single-GPU exhaustive baseline](B200_E1_PLAN.md)
- [E2: four-GPU exhaustive baseline](B200_E2_PLAN.md)
- [E3: memory-default TileTune evaluation](B200_E3_PLAN.md)

This file retains the shared matrix, workload inventory, and combined-report
contract. Run completion status belongs to each frozen manifest and its
`completed.json` markers rather than to this protocol document. The runner is
`experiments/common/b200.py`. The grouped-compilation failure recovery required
by E3 is implemented in `tilelang/autotuner/grouped_compile.py` and covered by
`testing/python/autotune/test_grouped_compile_fallback.py`.

A live one-GPU B200 smoke under the final shared 64-worker compiler setting
passed for both exhaustive and E3 execution under the former CUPTI protocol:
the E3 smoke analyzed the then-current 1,473-config GEMM pool, froze a strict alpha
selection, compiled eight candidates as one group, observed complete zero-spill
PTXAS counters, and checked correctness. That smoke is implementation evidence
only; the change to CUDA-event timing requires fresh one- and four-GPU event
preflights before the final run. Only one B200 was idle during implementation
validation.

## Frozen experiment matrix

Run all five kernel families and all 25 final workloads in E1, E2, and E3: 75
accepted workload runs. Launch E1, E2, and E3 as three separate, dependency-
ordered Slurm jobs, each with its own output root and frozen manifest. Each job
processes its 25 workloads sequentially. The scheduler may allocate different
physical NVIDIA B200s to the three jobs or to a job after requeue; record every
attempt's UUIDs and treat device-to-device variance as a limitation of the
comparison. Resume requires the same GPU count, model, and compute capability,
but does not require the same physical UUIDs. The frozen CPU IDs must still be
available.
The final inventory is `experiments.common.spec.default_workloads(smoke=False)`,
which resolves each family's `cases(holdout=True)` definitions.

| Experiment | Candidate selection | PTXAS resource handling | Compiler workers | Event benchmark workers | Compile/benchmark pipeline | Grouped compilation | TileTune |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| E1 | Exhaustive full pool | Report only | 64 shared | 1 worker / 1 GPU | Off | Off | Off |
| E2 | Exhaustive full pool | Report only | 64 shared | 4 workers / 4 GPUs | Off | Off | Off |
| E3 | Full-pool analysis, strict memory alpha=0.5 | Reject above oracle-calibrated family limits | 64 shared | 4 workers / 4 GPUs | On | On, groups of 8 | On |

`multigpu=4` means that candidate measurements for one workload are distributed
over four GPUs. It does not mean four workloads execute concurrently. E1 asks
Slurm for one B200; E2 and E3 ask for four B200s on one node. Compilation and
benchmarking have independent worker controls: every workload owns exactly one
shared 64-worker compiler pool, and the benchmark stage owns one or four GPU
workers. The 64 compiler workers are not divided by GPU count and are never
instantiated once per benchmark GPU. E3 may overlap these separately
provisioned stages, but still has only one compiler pool. Record the resolved
compiler count and benchmark worker/device count.

E1 and E2 attempt every config. Every successfully compiled E1/E2 kernel proceeds
to independent correctness checking and CUDA-event benchmarking; PTXAS spill and
local-memory observations never prune the exhaustive oracle. Before E3 starts,
the runner checks every exact E1/E2 minimum, including latency ties, against the
family limits below. E3 then enforces those limits after compilation. E3 elaborates and analyzes the complete original
pool, uses `ranking_metric="memory"`, and applies `alpha=0.5`. Selection uses
`floor(pool_size * 0.5)` and admits only complete equal-score groups whose
conservative tail rank fits within that budget. It must not split a tie, expand
past the half-pool limit, or replace a candidate after analysis, compilation,
post-compile, correctness, or benchmark failure.

Keep the existing B200 memory ordering unchanged: lexicographic `(B, -D, E)`,
where `B` is logical global byte-waves, `D` is the IR pipeline depth, and `E`
is logical access-waves. Its exact integer encoding is
`65535*B*(B+1)//2 + (65535-D)*(B+1) + E`. Do not force the H200 `G,e`
extension into this run. The compiler-only family policy below is independent
of that score.

All experiments use `early_stop=False`. E1/E2 use neither compile/benchmark
pipelining nor grouped compilation. E3 uses both pipelining and grouped
compilation with `group_compile_size=8`.

## Shared measurement and validity settings

- Benchmark backend: `event`. The saved candidate latency is the arithmetic mean
  of per-repetition CUDA-event elapsed times. Each pair of events brackets only
  the candidate invocation; the 256 MiB L2 flush is enqueued before the start
  event. An initial CUDA-event estimate determines the warmup and repeat counts.
- Event execution uses one benchmark worker per active GPU, with CUDA events,
  streams, cache buffers, and synchronization scoped to that worker's assigned
  device. The required four-GPU preflight must pass before accepting E2/E3
  measurements. Event timing is the only accepted backend; CUPTI is not a
  fallback.
- Warmup budget: 10 ms. Measurement budget: 50 ms. Candidate timeout: 60 s.
  L2 cache flush: 256 MiB outside the reported kernel time.
- Input seed: 123. Disable TF32. Use identical prepared inputs, references,
  tolerances, compiler target, example kernel, and pool ordering in E1/E2/E3.
- Set `TILELANG_DISABLE_CACHE=1` and
  `TILELANG_AUTO_TUNING_DISABLE_CACHE=1`. Use fresh worker processes and new
  output directories. Record Python, TileLang/native-build, TVM, Torch, CUDA,
  driver, NVCC, host compiler, source hashes, GPU UUIDs, clocks, and config IDs.
- Set an aggregate compiler limit of 64 and record the effective value. Prevent
  nested OpenMP/MKL/OpenBLAS/NumExpr/TVM pools by setting their thread counts to
  one. Select at least 72 visible logical CPUs for the worker: 64 for the shared
  compiler pool plus eight for benchmark/runtime headroom. At least one
  additional visible CPU must remain for the coordinator and monitor. Freeze the
  selected CPU IDs within each Slurm job; keep the count and selection policy
  identical across E1/E2/E3.
- Every candidate that reaches execution receives an independent numerical
  correctness check. Compilation, correctness, timeout, and benchmark failures
  are terminal recorded outcomes and are never silently dropped. Compiler-resource
  findings are report-only in E1/E2. E3 may reject a selected candidate only
  when exact PTXAS spill/local counters exceed the frozen limit for the
  workload's declared operation. Heuristic kernel classifications never select
  the limit.
- Hold a host lease and leases for the current allocation's GPU set for the
  whole coordinator run. Monitor the active subset continuously: one GPU for E1 and all four GPUs
  for E2/E3. Poll GPU state and compute processes once per second; a foreign GPU
  process or a monitor gap longer than five seconds invalidates the whole
  workload attempt. The polling guarantee cannot exclude subsecond overlap.
- Sample the selected CPU set on the same polling loop. Record external busy
  cores after subtracting the worker process group, I/O-wait cores, and steal
  cores in `gpu_observations.jsonl`; summarize their maxima and whether they
  exceed 8, 1, and 0.1 cores respectively in `monitor.json`. CPU contention is
  telemetry-only on this shared host and does not restart the workload. Report
  compilation, XGBoost, and end-to-end timing as shared-host measurements.
  Preserve and retry GPU-contended or monitor-gap attempts from the beginning;
  do not splice partial timing records.

### CPU placement and monitored launch flow

1. Each mode coordinator takes one host-wide advisory lease plus a lease for
   every GPU in its allocation, so a second cooperating experiment cannot start
   on either the same host or a supposedly disjoint leased GPU set.
2. For a new output root, it samples `/proc/stat` for one second, ranks physical
   core sibling groups by activity, and selects at least 72 logical CPUs: 64
   compiler workers plus eight logical CPUs of runtime/benchmark headroom. At
   least one additional visible CPU must remain for the coordinator and monitor.
   The exact worker CPU list, initial allocation's GPU inventory,
   source/native-build identity, interpreter, requests, and event backend are
   frozen in `manifest.json`. A resumed allocation may use different UUIDs only
   when its GPU count, model, and compute capability match.
3. The coordinator moves itself to CPUs outside the selected worker set. Every
   fresh workload child is pinned to the frozen set before importing Torch. The
   child receives only the active GPU UUIDs in `CUDA_VISIBLE_DEVICES`, cache
   disabling variables, both TileLang compiler limits set to 64, and
   OpenMP/MKL/OpenBLAS/NumExpr/TVM thread counts set to one.
4. The monitor waits for five acceptable one-second samples before spawning the
   child: every active GPU has at most 5% utilization, no foreign compute
   process is present, and polling is timely. It captures combined stdout and
   stderr in `worker.log`, per-poll GPU/CPU observations in
   `gpu_observations.jsonl`, and the final validity decision in `monitor.json`.
   `attempt.json` records the Slurm cluster, job, partition/QOS, node list,
   restart count, and execution node when those variables are available.
5. Workloads run sequentially within a mode. Slurm `afterok` dependencies order
   the mode jobs E1, then E2, then E3. E1 exposes one GPU; E2/E3 expose four and
   distribute one workload's candidate benchmarks across them. The launcher
   never runs multiple workloads concurrently. GPU contention or a monitor gap
   kills the whole child process group, preserves the attempt directory, and
   retries from a fresh attempt up to three times.
6. Only a terminal, GPU-uncontended attempt with the frozen backend, exact
   CPU/GPU worker counts, complete candidate outcomes, and a valid measured
   winner gets `completed.json`. Resume revalidates hashes and identities before
   reusing it. `--resume-or-start` safely distinguishes a new root, a frozen
   resumable root, and an interrupted manifest initialization. Results created
   under the former CUPTI setting therefore cannot be resumed into this
   event-timed study.

### Oracle-safe post-compile resource policy

Turn exact PTXAS resource capture on in all three experiments. E1 and E2 use
report-only capture with no spill/local limit so the exhaustive measurements
define their oracles without compiler-resource censoring. E3 keeps the memory
analysis configuration at `mode="report_only"`, `max_spill_bytes=None`, and
`max_local_bytes=None`, then supplies a separate reject-mode
`post_compile_policy` with the following operation-derived limits. This keeps
the compiler gate out of analysis, scoring, and selection:

| Family | Audited E1 cases | Largest oracle spill store/load | Largest oracle local | E3 spill limit | E3 local limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 GEMM | 5/5 | 0 / 0 B | 0 B | 0 B | 0 B |
| FlashAttention | 5/5 | 0 / 0 B | 0 B | 0 B | 0 B |
| KDA intra | 5/5 | 0 / 0 B | 0 B | 0 B | 0 B |
| FP8 GEMM | 5/5 | 12 / 8 B | 8 B | 16 B | 16 B |
| Grouped GEMM | 5/5 | 0 / 0 B | 0 B | 0 B | 0 B |

The FP8 cap rounds the largest observed allocation up to 16 bytes, matching the
round-up convention used by the H200 policy while keeping the B200 observation
and policy independent. It is a real rejection boundary: a candidate at 17
bytes fails. The other families remain at zero because every measured oracle in
each of their five cases has complete zero counters; they are not put in report
mode merely because FP8 needs an allowance.

The individual E1 observations are:

| Case | Oracle index | Registers | Spill store/load | Local |
| --- | ---: | ---: | ---: | ---: |
| `gemm_decode` | 1472 | 168 | 0 / 0 B | 0 B |
| `gemm_prefill` | 957 | 240 | 0 / 0 B | 0 B |
| `gemm_ffn_down` | 957 | 240 | 0 / 0 B | 0 B |
| `gemm_square` | 1467 | 168 | 0 / 0 B | 0 B |
| `gemm_square_large` | 1466 | 168 | 0 / 0 B | 0 B |
| `attention_short_causal` | 501 | 168 | 0 / 0 B | 0 B |
| `attention_batched_causal` | 501 | 168 | 0 / 0 B | 0 B |
| `attention_noncausal` | 497 | 168 | 0 / 0 B | 0 B |
| `attention_causal` | 497 | 168 | 0 / 0 B | 0 B |
| `attention_long_causal` | 497 | 168 | 0 / 0 B | 0 B |
| `kda_intra_short` | 228 | 40 | 0 / 0 B | 0 B |
| `kda_intra_medium` | 228 | 40 | 0 / 0 B | 0 B |
| `kda_intra_regular` | 228 | 40 | 0 / 0 B | 0 B |
| `kda_intra_batched` | 228 | 40 | 0 / 0 B | 0 B |
| `kda_intra_long` | 228 | 40 | 0 / 0 B | 0 B |
| `gemm_fp8_decode` | 4 | 211 | 0 / 0 B | 0 B |
| `gemm_fp8_prefill` | 386 | 255 | 0 / 0 B | 0 B |
| `gemm_fp8_ffn_down` | 359 | 255 | 0 / 0 B | 0 B |
| `gemm_fp8_square` | 365 | 255 | 12 / 8 B | 8 B |
| `gemm_fp8_square_large` | 365 | 255 | 12 / 8 B | 8 B |
| `grouped_gemm_decode` | 190 | 240 | 0 / 0 B | 0 B |
| `grouped_gemm_prefill` | 382 | 240 | 0 / 0 B | 0 B |
| `grouped_gemm_aligned` | 383 | 168 | 0 / 0 B | 0 B |
| `grouped_gemm_down_aligned` | 524 | 240 | 0 / 0 B | 0 B |
| `grouped_gemm_ragged` | 567 | 168 | 0 / 0 B | 0 B |

The legacy heuristic classifications are diagnostic only. In particular, the
B200 attention kernels can be labeled `quantized_gemm`, and FP8 GEMM can be
labeled `linear_attention`; neither label is allowed to choose the cap. The
frozen workload operation (`attention`, `gemm_fp8`, and so on) selects it, while
the decision itself uses exact PTXAS counters.

Immediately before E3, `--audit-resource-policy` loads every exact minimum and
latency tie from both E1 and E2, requires a post-compile row with complete
register/spill/local counters, and checks it against the declared family cap.
It writes `oracle_resource_policy.json` and `.csv`. E3 does not launch if any of
the 50 workload/mode oracle sets is missing, has incomplete counters, or exceeds
the cap. If E2 reveals a larger legitimate requirement, stop, update the family
policy, and restart the three-mode chain so all manifests share the revised
source identity; never let E3 silently filter that oracle.

## Kernel inventory

| Family | Experiment adapter | Authoritative kernel source | Operation and dtype | Pool/workload |
| --- | --- | --- | --- | ---: |
| GEMM | `experiments/gemm/kernel.py` | `examples/gemm_sm100/gemm_tcgen5mma.py::matmul` | SM100 TCGen05, `C=A@B.T`; BF16 inputs/output, FP32 accumulation | 609 |
| FlashAttention | `experiments/flash_attention/kernel.py` | `examples/flash_attention_sm100/mha_fwd_bshd.py::flashattn` | SM100 TCGen05/TMEM BSHD forward; BF16 Q/K/V/O, FP32 softmax/accumulation | 520 |
| KDA intra | `experiments/kda/kernel.py` | `examples/kda/chunk_intra_token_parallel.py::tilelang_chunk_kda_fwd_intra_token_parallel` | Token-parallel coefficient stage; BF16 Q/K/beta/outputs and FP32 gates/accumulation | 513 |
| FP8 GEMM | `experiments/gemm_fp8/kernel.py` | `examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py` | SM100 two-CTA/persistent TCGen05; E4M3 A/B, packed UE8M0 scales, FP32 accumulation, BF16 C | 533 |
| Grouped GEMM | `experiments/grouped_gemm/kernel.py` | `examples/grouped_gemm/example_grouped_gemm_fwd.py::grouped_gemm` | Packed concatenated BF16 grouped GEMM with FP32 accumulation | 576 |

The separate SM100 grouped-MXFP8 example is not substituted for grouped GEMM;
it has different input, scale, dtype, and output semantics.

## Final workload inventory

All listed counts are per workload. E1 and E2 each attempt 13,755 candidates.
E3 analyzes all 13,755 and can select at most 6,870 before tie-boundary
exclusions and failures. Across E1/E2/E3, at most 34,380 original candidate
slots are submitted for compilation; grouped-build retries are additional work
and must remain in end-to-end timing.

### GEMM — BF16, pool 609

| Workload | M | N | K | B layout |
| --- | ---: | ---: | ---: | --- |
| `gemm_decode` | 256 | 4,096 | 4,096 | `(N,K)`, transposed by GEMM |
| `gemm_prefill` | 1,024 | 4,096 | 4,096 | `(N,K)`, transposed by GEMM |
| `gemm_ffn_down` | 1,024 | 4,096 | 14,336 | `(N,K)`, transposed by GEMM |
| `gemm_square` | 4,096 | 4,096 | 4,096 | `(N,K)`, transposed by GEMM |
| `gemm_square_large` | 4,096 | 14,336 | 4,096 | `(N,K)`, transposed by GEMM |

E3 strict maximum: 304 configs per workload.

### FlashAttention — BF16 BSHD, pool 520

| Workload | Batch | Heads | Sequence | Head dim | Causal |
| --- | ---: | ---: | ---: | ---: | --- |
| `attention_short_causal` | 1 | 32 | 512 | 64 | Yes |
| `attention_batched_causal` | 2 | 16 | 2,048 | 64 | Yes |
| `attention_noncausal` | 1 | 32 | 4,096 | 128 | No |
| `attention_causal` | 1 | 32 | 4,096 | 128 | Yes |
| `attention_long_causal` | 1 | 16 | 8,192 | 128 | Yes |

E3 strict maximum: 260 configs per workload.

### KDA intra-token-parallel — BF16/FP32, pool 513

Every case fixes `dim=128`, `chunk_size=64`, and `sub_chunk_size=16`. Q/K/GK
have shape `(B,S,H,128)`, beta `(B,S,H)`, Aqk `(B,S,H,64)`, and Akk
`(B,S,H,16)`.

| Workload | B | S | H |
| --- | ---: | ---: | ---: |
| `kda_intra_short` | 1 | 2,048 | 32 |
| `kda_intra_medium` | 1 | 4,096 | 64 |
| `kda_intra_regular` | 1 | 8,192 | 32 |
| `kda_intra_batched` | 2 | 4,096 | 32 |
| `kda_intra_long` | 1 | 16,384 | 64 |

E3 strict maximum: 256 configs per workload.

### FP8 GEMM — E4M3 plus packed UE8M0 scales, pool 533

| Workload | M | N | K | Output |
| --- | ---: | ---: | ---: | --- |
| `gemm_fp8_decode` | 256 | 4,096 | 4,096 | BF16 |
| `gemm_fp8_prefill` | 1,024 | 4,096 | 4,096 | BF16 |
| `gemm_fp8_ffn_down` | 1,024 | 4,096 | 14,336 | BF16 |
| `gemm_fp8_square` | 4,096 | 4,096 | 4,096 | BF16 |
| `gemm_fp8_square_large` | 4,096 | 14,336 | 4,096 | BF16 |

All use `B=(N,K)`/`transpose_b=True` and scale granularity K=128. E3 strict
maximum: 266 configs per workload.

### Grouped GEMM — BF16, pool 576

`A=(sum(M_i),K)`. B is `(G,K,N)` when `transpose_b=False` and `(G,N,K)` when
`transpose_b=True`. C concatenates the group outputs along M.

| Workload | Group row counts | N | K | Transpose B |
| --- | --- | ---: | ---: | --- |
| `grouped_gemm_decode` | 1, 2, 4, 8 | 2,048 | 7,168 | No |
| `grouped_gemm_prefill` | 32 x 8 groups | 2,048 | 7,168 | No |
| `grouped_gemm_aligned` | 128 x 4 groups | 2,048 | 7,168 | No |
| `grouped_gemm_down_aligned` | 256 x 3 groups | 7,168 | 2,048 | Yes |
| `grouped_gemm_ragged` | 63, 77, 111, 280 | 7,168 | 2,048 | Yes |

E3 strict maximum: 288 configs per workload.

## Grouped-compilation failure recovery

E3 groups at most eight selected configs that share effective compiler settings.
Elaboration and lowering failures remain attributed to the individual config.
If merged device codegen, host codegen, module import, or executable JIT fails,
the unfinished group is bisected recursively down to singleton builds. A valid
neighbor therefore still reaches correctness and CUDA-event benchmarking.

Already completed configs and post-compile rejections are not retried. Fallback
never adds configs outside the frozen alpha selection. Each fallback group and
error is recorded in `grouped_compile_fallbacks`; failed shared attempts and
all retries count toward compile work and accepted end-to-end wall time.

## XGBoost end-to-end accounting

XGBoost is an accompanying selection baseline, not a fourth E1/E2/E3 execution
mode. Its primary end-to-end number must include all of the following:

1. CUDA-event collection and correctness checking for the fixed 10% samples from
   both training workloads and the validation workload of each family.
2. Feature/schema construction, DMatrix construction, model fitting, validation,
   and early stopping (600-round cap, depth 10, eta 0.05, subsample 0.8,
   patience 20, seed 123).
3. Model serialization/loading and held-out prediction/ranking.
4. Compilation, report-only post-compile resource capture, correctness, and
   CUDA-event benchmarking of selected final candidates.

Use the existing two training and one validation workload per family from each
`cases.py`; none overlaps a final shape. Record
`training_collection_seconds`, `validation_collection_seconds`, `fit_seconds`,
model load/prediction time, and selected-candidate online time separately. The
suite-level XGBoost end-to-end total charges each family model's collection and
training once, plus all 25 online final-workload costs. A reused frozen model
must carry its original recorded collection/training cost; it is never reported
as free. Oracle sweeps and post-run winner remeasurement are excluded from the
method's end-to-end time.

## Runner and preflight gates

Use `run_b200_slurm.sh` as the batch entry point. It starts or resumes exactly
one mode, forwards termination to the coordinator so the active attempt is
marked interrupted, and asks Slurm to requeue at the wall-time warning. E1/E2
completion markers are preserved across later jobs. After E3 completes, the
same script validates all 75 completed workloads and writes the combined
comparison and oracle-retention reports.

```bash
# Standard-library-only plan; no CUDA query and no output creation.
python -m experiments.common.b200 --plan

# Site-specific values: adjust these four settings for the cluster.
export B200_SLURM_PARTITION=preempt
export B200_SLURM_QOS=preempt
export B200_SLURM_CONSTRAINT=b200
export B200_SLURM_TIME=24:00:00

b200_run_id="$(date -u +%Y%m%dT%H%M%SZ)"
b200_study_root="experiments/results/b200-final-three-${b200_run_id}"
mkdir -p "${b200_study_root}/slurm"

slurm_common=(
  --partition="${B200_SLURM_PARTITION}"
  --qos="${B200_SLURM_QOS}"
  --constraint="${B200_SLURM_CONSTRAINT}"
  --nodes=1
  --ntasks=1
  --cpus-per-task=80
  --time="${B200_SLURM_TIME}"
)
launcher=experiments/common/run_b200_slurm.sh

e1_job="$(sbatch --parsable "${slurm_common[@]}" --gpus=1 \
  --job-name=b200-e1 --output="${b200_study_root}/slurm/%x-%j.log" \
  "${launcher}" E1 "${b200_study_root}")"

e2_job="$(sbatch --parsable "${slurm_common[@]}" --gpus=4 \
  --dependency="afterok:${e1_job}" --job-name=b200-e2 \
  --output="${b200_study_root}/slurm/%x-%j.log" \
  "${launcher}" E2 "${b200_study_root}")"

e3_job="$(sbatch --parsable "${slurm_common[@]}" --gpus=4 \
  --dependency="afterok:${e2_job}" --job-name=b200-e3 \
  --output="${b200_study_root}/slurm/%x-%j.log" \
  "${launcher}" E3 "${b200_study_root}")"

printf 'E1=%s E2=%s E3=%s root=%s\n' \
  "${e1_job}" "${e2_job}" "${e3_job}" "${b200_study_root}"
```

If the site uses typed GRES rather than `--gpus`, replace those options with its
equivalent, for example `--gres=gpu:b200:1` and `--gres=gpu:b200:4`. Add the
site's account/reservation flags to `slurm_common` as needed. Submit from the
repository root; the output root must be on storage visible after requeue.
Activate the project environment before submission, or export `PYTHON` as the
absolute path of its Python executable so every restart uses the same
interpreter.

Each mode automatically performs its five-family preflight under
`${b200_study_root}/MODE/preflight`, giving 15 preflight workloads across the
three jobs. `--resume-or-start` revalidates completed artifacts and restarts only
the interrupted workload from a new attempt directory. Slurm requeue starts the
batch script again with the same job ID. The script's `USR1` wall-time handler
requests requeue after graceful cleanup; scheduler-initiated preemption still
depends on the site's `PreemptMode`/`GraceTime` configuration.
See Slurm's [`sbatch` requeue/signal contract](https://slurm.schedmd.com/sbatch.html)
and [preemption modes](https://slurm.schedmd.com/preempt.html) when adapting the
template to a site policy.

Resume accepts replacement physical GPU UUIDs when the allocation has the same
GPU count, B200 model, and compute capability. Each workload attempt records the
UUIDs actually used in `monitor.json`, and the combined comparison reports those
per-attempt UUIDs. Resume still rejects an allocation without the frozen CPU IDs
or with a different GPU class. Cross-device continuation is therefore explicit
and auditable rather than silent.

The final E3 step runs the equivalent of:

```bash
python -m experiments.common.b200 --combine "${b200_study_root}"
```

It requires complete `E1`, `E2`, and `E3` subdirectories with matching source,
interpreter, revision, pools, and event settings. It writes `comparison.json`,
`oracle_retention.json`, and `oracle_retention.csv` in the study root.

The current `experiments.combined` runner is not a substitute: it is hard-coded
for two GPUs, four-config compile groups, event timing, four compiler workers,
and report-only spill counters.

Before the three-job chain, all of these gates must pass:

1. Offline plan asserts exactly 25 unique final workloads, the five sources
   above, the frozen pool counts/IDs, E1/E2/E3 flags, aggregate workers=64,
   event timing, report-only resource capture in E1/E2, the per-family strict
   E3 resource policy,
   group size 8, and strict alpha budgets.
2. Group fallback tests pass, including mixed valid/invalid device and host
   builds, spill rejection, no selection refill, and failed-attempt accounting.
3. One-process CUDA-event smoke succeeds for each family.
4. Four device-scoped CUDA-event benchmark workers run concurrently without a
   crash, cross-device attribution, or missing samples. Any worker crash or
   incomplete outcome invalidates the attempt; CUPTI is never used as a
   fallback.
5. Each family passes E1/E2/E3 preflight with identical config IDs, numerical
   results, exact compiler-worker/GPU counts, resource evidence, and terminal
   outcome rows. E1/E2 may contain no post-compile rejection. E3 rejections are
   permitted only for exact counters above the declared family limit.
6. Resume validation rejects any change in measurement-source/native-build
   hash, workload, pool/order, settings, benchmark backend, GPU count/model/
   compute capability, or spill policy. Physical GPU UUID changes are accepted
   and recorded per attempt.
7. After E2 and before E3, the resource-policy audit covers all 50 E1/E2
   workload/mode oracle sets and every exact latency tie; every row has complete
   counters and passes its family limit.

## Oracle-retention audit

For every workload, independently find all minimum-latency valid configs in E1
and E2. A valid oracle candidate must have compiled, passed numerical
correctness, and completed CUDA-event measurement. E1/E2 resource findings do
not remove a measured candidate from the exhaustive oracle. The pre-E3 audit
requires complete PTXAS counters and proves that each oracle fits its declared
family limit. Take the union of E1 and E2 oracle config IDs; never redefine the
oracle after looking at E3.

For every oracle config, require E3 to show:

1. Identical workload, source identity, ordered pool, config, and config ID.
2. A finite eligible memory score and conservative equal-score tail rank no
   greater than `floor(pool_size / 2)`.
3. Membership in the frozen E3 selected indices.
4. Successful compilation after any recorded group fallback.
5. A complete post-compile record with `keep=true`, `status="pass"`, and exact
   spill/local counters within the frozen family limit.
6. Passing numerical correctness and a completed CUDA-event benchmark.

Write `oracle_retention.json` and `oracle_retention.csv` with workload, E1/E2
origin, config ID/config, oracle CUDA-event latency, pool size, alpha budget,
memory score, tail rank/fraction, selection membership, fallback history,
compiler resources, post-compile decision, correctness, benchmark status, and
reason.

Final success requires every E1/E2 oracle config for all 25 workloads to pass
the identity, rank, selection, compilation, family resource, correctness, and
measurement gates, and no E3 selected set may exceed half of its original pool.
A failure is preserved and diagnosed; do not change alpha, reorder the pool, or
substitute another config.

## Timing and result reporting

Accepted tuning time is elapsed wall time from cold workload-worker start until
its complete terminal outcome set and winner are durable. E3 includes full-pool
elaboration/analysis, selection, all original and fallback compilation work,
correctness, and CUDA-event benchmarking. E1/E2 include all exhaustive compilation,
resource capture, correctness, and CUDA-event benchmarking. Queue/idle wait,
contaminated discarded attempts, preflight, oracle audit, and seven-repeat
winner verification are reported separately.

Report per workload, family, and total:

- accepted worker wall time and tuning wall time;
- E1/E2 and E1/E3 speedups;
- analysis, compilation, fallback, correctness, and CUDA-event benchmark breakdowns;
- attempted/compiled/resource-observed/correct/benchmarked counts, plus observed
  spill/local, missing-counter, and family-limit rejection counts for E3;
- selected count and alpha shortfall for E3;
- winner config/ID, sweep latency, and seven fresh single-GPU CUDA-event samples;
- arithmetic/geometric mean per-case speedup and total-time speedup;
- XGBoost training-inclusive end-to-end time and its collection/fit/online
  breakdown.

Do not sum overlapping pipeline stage work and call it elapsed time. Preserve
both summed work and actual wall time. E2 versus E1 measures multi-GPU candidate
distribution. E3 versus E2 measures the combined effect of TileTune selection,
pipeline overlap, and grouped compilation; this three-run matrix does not
attribute those three E3 effects independently.

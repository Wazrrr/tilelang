# B200 E1 plan: single-GPU exhaustive baseline

## Purpose

E1 measures the complete candidate pool on one NVIDIA B200. It establishes the
single-GPU end-to-end baseline, an unfiltered measured oracle for every workload,
and exact PTXAS resource evidence for calibrating E3. TileTune selection is off;
the library-wide TileTune default is `memory`, but no analytical ranking may
remove or reorder E1 candidates.

## Frozen scope

| Family | Final workloads | Pool per workload | E1 candidates |
| --- | ---: | ---: | ---: |
| BF16 GEMM | 5 | 609 | 3,045 |
| FlashAttention | 5 | 520 | 2,600 |
| KDA intra | 5 | 513 | 2,565 |
| FP8 GEMM | 5 | 533 | 2,665 |
| Grouped GEMM | 5 | 576 | 2,880 |
| **Total** | **25** | — | **13,755** |

The exact workloads come from
`experiments.common.spec.default_workloads(smoke=False)`. The ordered pools and
config IDs must match E2 and E3 exactly.

## Execution configuration

- Variant: `baseline` / experiment `E1`.
- GPU allocation: one B200 and one CUDA-event benchmark worker.
- Workload concurrency: one; all 25 workloads run sequentially.
- Compiler pool: one shared pool of exactly 64 workers.
- Compile/benchmark overlap: disabled.
- Grouped compilation: disabled.
- Candidate policy: exhaustive; attempt every one of the 13,755 slots.
- Timing: CUDA events only, 10 ms warmup, 50 ms measurement, 60 s candidate
  timeout, and a 256 MiB L2 flush outside the measured event interval.
- Seed: 123; TF32 disabled; fresh worker process and cold TileLang/autotune
  caches for each workload.
- CPU placement: at least 72 logical CPUs in the frozen worker set, leaving at
  least one additional CPU for the coordinator/monitor. Nested numerical and
  compiler thread pools are fixed to one.

## Resource and oracle contract

PTXAS capture is enabled for every successfully compiled candidate, but the
filter action is report-only with no spill or local-memory limit. A spill,
local allocation, or surprising heuristic kernel classification is evidence,
not a reason to suppress correctness checking or benchmarking.

For each workload:

1. Elaborate and compile every candidate independently.
2. Record exact register, spill-store, spill-load, and local-memory counters.
3. Run the workload-specific numerical check for every compiled candidate.
4. CUDA-event benchmark every correct candidate.
5. Record every compile, correctness, timeout, and benchmark failure as a
   terminal outcome; never silently drop or replace a row.
6. Define the E1 oracle as every correct, benchmarked candidate tied at the
   minimum saved latency. Do not use a resource limit to redefine that set.

An accepted E1 workload may contain compile or correctness failures, but it may
not contain `post_compile_rejected`, an incomplete outcome, or a missing valid
winner.

## Isolation and restart flow

The coordinator holds a host lease and its one-GPU lease for the whole run. It
waits for five clean one-second GPU samples before each child, monitors the GPU
and selected CPUs once per second, and invalidates an attempt for a foreign GPU
process or a monitor gap longer than five seconds. CPU contention remains
recorded telemetry. A contaminated attempt is preserved and restarted from the
beginning; completed workloads remain reusable after manifest/hash validation.

Before the full sweep, the runner performs one eight-candidate preflight for
each of the five families. Only a GPU-uncontended attempt with the exact worker
counts, event backend, complete terminal outcomes, matching config IDs, and a
measured winner receives `completed.json`.

## Launch

Slurm allocates the one visible B200 and the launcher writes under `E1`:

```bash
launcher=experiments/common/run_b200_slurm.sh
e1_job="$(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=80 \
  --gpus=1 --job-name=b200-e1 \
  --output="${b200_study_root}/slurm/%x-%j.log" \
  "${launcher}" E1 "${b200_study_root}")"
```

For a direct run on an already allocated host where physical GPU 6 is confirmed
idle:

```bash
python -m experiments.common.b200 \
  --experiments E1 --gpus 6 \
  --output "${b200_study_root}/E1" --resume-or-start
```

## Required outputs and acceptance

Each workload must have a frozen request, environment/source identity,
`outcomes.json`, `compilation.json`, `resource-filter.tsv`, `benchmarks.tsv`,
monitor record, summary, attempt record, and hashed completion marker. E1 is
complete only when all 25 markers validate.

Report per workload and by family:

- accepted worker wall time and tuning time;
- compile/correctness/benchmark counts and failure counts;
- measured winner and every exact minimum-latency tie;
- PTXAS register/spill/local observations for each oracle;
- GPU UUID and recorded host/GPU isolation telemetry.

E1 does not claim TileTune effectiveness. Its output is the single-GPU
exhaustive reference consumed by the E2 comparison and the pre-E3 oracle audit.

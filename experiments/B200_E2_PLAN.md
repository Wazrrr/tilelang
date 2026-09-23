# B200 E2 plan: four-GPU exhaustive baseline

## Purpose

E2 repeats the exact E1 exhaustive work while distributing one workload's
candidate benchmarks across four B200s. It isolates multi-GPU benchmark-stage
scaling from TileTune selection, compile/benchmark pipelining, and grouped
compilation. E2 also supplies an independent unfiltered oracle set for E3.

## Frozen scope

| Family | Final workloads | Pool per workload | E2 candidates |
| --- | ---: | ---: | ---: |
| BF16 GEMM | 5 | 1,473 | 7,365 |
| FlashAttention | 5 | 520 | 2,600 |
| KDA intra | 5 | 513 | 2,565 |
| FP8 GEMM | 5 | 533 | 2,665 |
| Grouped GEMM | 5 | 576 | 2,880 |
| **Total** | **25** | — | **18,075** |

For every workload, the ordered config dictionaries and config IDs must equal
E1 position by position. A different physical B200 allocation is allowed, but
the GPU count, model, compute capability, measurement backend, source identity,
native build, and candidate pools remain frozen.

## Execution configuration

- Variant: `multi_gpu` / experiment `E2`.
- GPU allocation: four B200s and four device-scoped CUDA-event workers.
- Distribution unit: candidates of one workload; never run four workloads in
  parallel.
- Compiler pool: one shared pool of exactly 64 workers—not 64 per GPU.
- Compile/benchmark overlap: disabled.
- Grouped compilation: disabled.
- Candidate policy: exhaustive; attempt all 18,075 slots.
- Timing, seed, cache, correctness, CPU placement, and candidate timeout:
  identical to E1.

The four benchmark workers own separate devices, streams, events, cache-flush
buffers, input tensors, and synchronization. Cross-device attribution or a
missing worker result invalidates the attempt. CUPTI is not a fallback.

## Resource and oracle contract

E2 uses the same report-only PTXAS capture as E1: no spill/local-memory finding
may filter a compiled candidate. Every successfully compiled candidate must
proceed to correctness and, if correct, CUDA-event measurement.

For each workload, define the E2 oracle independently as all correct,
benchmarked candidates tied at its minimum saved latency. The E1 and E2 winner
sets are allowed to differ because they are independent measurements on
possibly different physical devices. This is reported as an oracle-match
result, not repaired by substituting one run's winner into the other.

The comparison must report:

1. exact ordered candidate-pool identity;
2. E1 and E2 minimum latency and all tied config IDs;
3. whether the oracle sets are identical, overlap, or disjoint;
4. tuning-time and accepted worker-wall-time speedup `E1 / E2`;
5. candidate status-count differences and resource-counter differences;
6. the GPU UUIDs used by each accepted attempt.

The pre-E3 reference is the union of the exact E1 and E2 minima, including all
ties. E2 must never be filtered merely to make its oracle agree with E1.

## Isolation and restart flow

The coordinator leases the host and all four GPUs. Five clean one-second samples
are required before each child; a foreign process on any active GPU or a monitor
gap longer than five seconds invalidates the whole workload attempt. CPU
contention is recorded but does not restart it. Workloads run sequentially,
and a failed or contaminated attempt is preserved before a fresh retry.

The automatic preflight runs one eight-candidate workload for each family using
all four event workers. It must demonstrate complete outcomes and correct
device attribution before the full sweep starts.

## Launch

E2 is a separate four-GPU job. An `afterok` dependency is recommended when E1
has not already completed:

```bash
launcher=experiments/common/run_b200_slurm.sh
e2_job="$(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=80 \
  --gpus=4 --dependency="afterok:${e1_job}" --job-name=b200-e2 \
  --output="${b200_study_root}/slurm/%x-%j.log" \
  "${launcher}" E2 "${b200_study_root}")"
```

If E1 is already validated, omit the scheduler dependency but retain the same
study root so the later audit can find `${b200_study_root}/E1` and `/E2`.

## Required outputs and acceptance

The required artifact set matches E1 and additionally proves four benchmark
devices were active. E2 is complete only when all 25 completion markers validate
and no outcome is `post_compile_rejected`.

E2 acceptance does not require a particular speedup or identical measured
winner. It requires a fair exhaustive comparison: identical work, four valid
device-scoped benchmark workers, complete unfiltered measurements, and explicit
reporting of both end-to-end scaling and oracle agreement.

# B200 E3 plan: memory-default TileTune evaluation

## Purpose

E3 measures the combined effect of TileTune memory selection, compile/benchmark
pipelining, grouped compilation, and four-GPU measurement against E1 and E2. It
must analyze the full original pool, select no more than half, preserve every
E1/E2 oracle, and enforce an independently audited post-compile resource policy.

`memory` is the default TileTune ranking method. E3 still records
`ranking_metric="memory"` explicitly in its frozen manifest so archived results
remain unambiguous if defaults change later.

## Memory ordering

Use the existing B200 ordering unchanged:

```text
(B, -D, E)
B = logical global byte-waves
D = IR pipeline depth
E = logical access-waves

score = 65535*B*(B+1)//2 + (65535-D)*(B+1) + E
```

This is the original B200 memory model. Do not force the H200 `G,e` extension
into E3. Equal `(B, -D, E)` triples remain one primary-score tie group.

## Frozen scope and selection budgets

| Family | Workloads | Pool/workload | Strict floor(pool/2) | Maximum selected |
| --- | ---: | ---: | ---: | ---: |
| BF16 GEMM | 5 | 1,473 | 736 | 3,680 |
| FlashAttention | 5 | 520 | 260 | 1,300 |
| KDA intra | 5 | 513 | 256 | 1,280 |
| FP8 GEMM | 5 | 533 | 266 | 1,330 |
| Grouped GEMM | 5 | 576 | 288 | 1,440 |
| **Total** | **25** | **18,075 analyzed** | — | **9,030 maximum** |

For each workload, TileTune elaborates and analyzes every original candidate.
Selection uses `alpha=0.5` with a strict budget: include an equal-score group
only if its conservative tail rank is at most `floor(pool_size/2)`. Never split
a tie, expand beyond the half-pool budget, or refill after an analysis,
compilation, resource, correctness, timeout, or benchmark failure.

## Mandatory pre-E3 oracle/resource audit

E3 must not start until both E1 and E2 contain 25 validated exhaustive results.
For each workload and each mode, independently find every correct, benchmarked
minimum-latency candidate, including exact ties. Require complete PTXAS register,
spill-store, spill-load, and local-memory counters for every oracle.

Apply the family policy selected by the workload's declared operation—not by a
heuristic kernel classification:

| Family | Maximum spill bytes | Maximum local bytes |
| --- | ---: | ---: |
| BF16 GEMM | 0 | 0 |
| FlashAttention | 0 | 0 |
| KDA intra | 0 | 0 |
| FP8 GEMM | 16 | 16 |
| Grouped GEMM | 0 | 0 |

The audit writes `oracle_resource_policy.json` and `.csv` and must cover all 50
workload/mode oracle sets plus every tied minimum. If any counter is missing or
an oracle exceeds its family cap, stop. Recalibrate that family from the actual
oracle evidence and restart the frozen chain; never let E3 silently filter the
oracle.

The analysis configuration remains report-only with no spill/local limit so
resource policy cannot change analysis, memory scoring, or selection. The
separate `post_compile_policy` applies the table above only after compilation,
using exact counters. A selected candidate exceeding its limit is recorded as
rejected and is not replaced.

## Execution configuration

- Variant: `tiletune` / experiment `E3`.
- GPU allocation: four B200s and four CUDA-event benchmark workers.
- Compiler pool: one shared pool of exactly 64 workers.
- Compile/benchmark pipeline: enabled.
- Grouped compilation: enabled, maximum group size eight among candidates with
  compatible effective compiler settings.
- Candidate timing/correctness, CPU placement, seed, cache state, and isolation:
  identical to E1/E2.
- Workload concurrency: one; pipeline stages may overlap only within that
  workload.

If a merged device build, host build, module import, or executable JIT fails,
recursively bisect the unfinished group to singleton builds. Preserve failed
group attempts and charge all fallback work to E3. Already completed candidates
and post-compile rejections are not retried, and fallback cannot add anything
outside the frozen selection.

## Launch

The E3 launcher first runs the mandatory E1/E2 resource audit, executes E3, then
combines all three modes and runs oracle retention:

```bash
launcher=experiments/common/run_b200_slurm.sh
e3_job="$(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=80 \
  --gpus=4 --dependency="afterok:${e2_job}" --job-name=b200-e3 \
  --output="${b200_study_root}/slurm/%x-%j.log" \
  "${launcher}" E3 "${b200_study_root}")"
```

The equivalent explicit gates are:

```bash
python -m experiments.common.b200 --audit-resource-policy "${b200_study_root}"
python -m experiments.common.b200 --experiments E3 \
  --output "${b200_study_root}/E3" --resume-or-start
python -m experiments.common.b200 --combine "${b200_study_root}"
```

## Oracle-retention acceptance

For every E1/E2 oracle config, E3 must prove all of the following:

1. identical workload, ordered pool, config, config ID, and source identity;
2. a finite eligible memory score;
3. conservative equal-score tail rank no greater than the half-pool budget;
4. membership in the frozen E3 selection;
5. successful compilation after any recorded group fallback;
6. complete exact PTXAS counters and a passing family post-compile decision;
7. passing numerical correctness and completed CUDA-event measurement.

`oracle_retention.json` and `.csv` preserve the origin mode, oracle latency,
memory score, rank/tie boundary, selection, fallback history, resources,
correctness, benchmark status, and failure reason. Final E3 success requires all
E1 and E2 minima—including ties—for all 25 workloads to pass every gate.

## Timing and effect report

E3 end-to-end time starts at the cold workload worker and includes full-pool
elaboration/analysis, selection, all original and fallback compilation work,
post-compile decisions, correctness, and CUDA-event benchmarking through durable
terminal outcomes. Preflight, queue delay, contaminated discarded attempts,
pre-E3 audit, and optional winner remeasurement are reported separately.

Report per workload, family, and total:

- E1/E3 and E2/E3 tuning-time and accepted wall-time speedups;
- analysis, compilation, fallback, correctness, and benchmark breakdowns;
- analyzed, selected, compiled, resource-rejected, correct, and benchmarked
  counts, including alpha shortfall from excluded boundary ties;
- oracle retention at rank, selection, compilation, resource, correctness, and
  benchmark stages;
- selected winner and fresh verification samples.

Because E3 changes selection, stage overlap, grouped compilation, and GPU count
relative to E1—and selection, overlap, and grouping relative to E2—the result is
a combined-system effect. Do not attribute the total speedup to the memory model
alone without an additional controlled ablation.

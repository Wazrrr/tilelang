**TileTune model audit: A100 experiment and GPU architecture coverage**

2026-09-14. Audited analysis version 18 on `dev`, base commit `113995b0afc8da1adaad009b1b896715b4d17176`. This extends the [A100 evaluation](tiletune_ampere_evaluation.md). At the time of this version-18 audit, every source file fingerprint recorded by the original experiment matched. The findings below describe that frozen model. The subsequent [Ampere pipeline revision](tiletune_ampere_pipeline.md) implements several corrections in version 20 and evaluates them separately; the original benchmark results are unchanged.

**The main selection loss comes from excluding useful schedules, but the model also contains concrete work-counting bugs and inaccurate resource estimates.** Fixing Ampere's stage gate alone would leave substantial defects. Recognizing a GPU target, compiling a kernel, producing a finite score, and predicting performance accurately are four separate capabilities.

The audit covers all 2,160 candidate-case pairs in the 34 hardware-supported held-out cases, including all 1,998 correct oracle measurements and 162 compilation failures. It also reviews TileTune's analysis, scheduling, resource, profiling and target code, and reproduces several missing-operation cases outside the benchmark suite. Only A100 runtime performance was measured. Other architectures below are code-coverage findings and validation requirements.

**Evidence and reproducibility.** The [candidate audit](../experiments/results/ampere-full-comparison-idle/model-audit.json) records every candidate's exact reasons, oracle outcome, coverage ceiling, selection loss and available compiler registers. The [minimal reproductions](../experiments/results/ampere-full-comparison-idle/model-repros.json) contain original IR and analysis results. The [source inventory](../experiments/results/ampere-full-comparison-idle/model-source-guards.json) records 121 explicit unknown/error/validation statements and source hashes; input validation is included in that inventory and is not itself a defect. Silent omissions require the separate findings below. The [work-count replay](../experiments/results/ampere-full-comparison-idle/model-count-counterfactual.json) changes only parallel-domain multipliers in memory, using saved profiles and reports.

Regenerate the candidate audit without a GPU or TileLang import:

```bash
python experiments/common/audit_model.py \
  experiments/results/ampere-full-comparison-idle \
  --output /tmp/tiletune-model-audit.json
```

Use a new output filename. The [audit script](../experiments/common/audit_model.py) checks that every candidate index and configuration matches the independent oracle. Diagnostic reproduction and replay scripts are retained beside the result files as `.py.txt` files.

**Measured coverage and selection loss.** Only 651/2,160 candidates receive eligible timing scores: 30.1% of the declared grid, or 32.6% of the correct, compilable candidates. There are 1,509 unknown scores and no `pressure_rejected` tiers in these timing reports. Only 3/34 exhaustive winners are eligible. Timing completes 26 cases; eight have no usable model.

| Workload family | Held-out cases | Eligible / full grid | Correct oracle candidates | Timing Oracle@K | Principal limitation |
| --- | ---: | ---: | ---: | --- | --- |
| GEMM: NN, NT, TN, batched, bias+ReLU, BF16, tall, wide | 16 | 576/1,728 | 1,728 | GM 0.818 | All stage-2/3 candidates unscored; every oracle winner uses those stages |
| Attention: FP16, causal FP16, BF16 | 6 | 48/324 | 162 | GM 0.732 | Positive stages; reduction layout restrictions; live-demand exclusion |
| Chunk KDA output | 2 | 16/48 | 48 | 1.000, 0.990 | Positive stages unscored; incomplete scalar cost and occupancy |
| Recurrent KDA | 2 | 0/12 | 12 | unavailable | Reduction ownership and direct global accesses in the recurrence |
| Softmax | 2 | 0/12 | 12 | unavailable | Reduction ownership; additional live-demand/occupancy restrictions |
| RMSNorm | 2 | 0/12 | 12 | unavailable | Reduction ownership and unsupported scalar intrinsic; additional resource restrictions |
| Reduce-sum | 2 | 0/12 | 12 | unavailable | Reduction ownership; an additional live-demand exclusion |
| Elementwise | 2 | 11/12 | 12 | 1.000, 0.988 | Scalar work undercount; one live-demand exclusion |

These results cover the portable adapters, not every TileLang example, backward pass or algorithm variant. CUDA FP8 and AMD FNUZ FP8 cases were declared hardware-unsupported on A100 and are outside the 34-case denominator.

For a correct eligible subset, the observed loss decomposes exactly as:

```text
Oracle@K = (global oracle latency / best eligible latency)
         × (best eligible latency / best frozen-shortlist latency)
             coverage ceiling          ranking within coverage
```

The second factor is **1.0 in all 16 GEMM and all six attention cases**. At the experiment's K, the shortlist already includes the best eligible candidate. Thus their current Oracle@K deficits are fully explained by the restricted eligible set. Attention's negative median rank correlation still matters: K=6 examines six of only eight eligible configurations, masking poor order within that small set. Better ordering alone cannot recover the excluded winners.

**Every observed timing-coverage reason.** Counts below deduplicate operation indices within each candidate. Reasons overlap; adding the rows would double-count candidates.

| Reason | Full-grid pairs | Correct oracle pairs | Affected cases and interpretation |
| --- | ---: | ---: | --- |
| Positive-stage scheduling unresolved | 1,400 | 1,292 | 1,152 GEMM, 216 attention, 32 chunk-KDA pairs; every positive stage in these grids |
| Reduction needs unresolved/inter-warp collective | 162 | 0 | 27/54 configurations in each attention case; these same candidates fail layout inference in the current compiler |
| No known fragment producer for reduction | 48 | 48 | Every recurrent-KDA, softmax, RMSNorm and reduce-sum configuration |
| Estimated live tile demand exceeds soft allowance | 28 | 28 | 18 attention, four RMSNorm, four softmax, one reduce-sum, one elementwise |
| Direct scalar global recurrence access unresolved | 12 | 12 | Every recurrent-KDA configuration |
| Unresolved operation work | 12 | 12 | Every RMSNorm configuration; its `rsqrt` expression has no supported cost |
| Unresolved wave count | 4 | 4 | Largest row tiles in RMSNorm and softmax at the larger test size; logical register proxy gives zero resident CTAs |

An additional generic message, `pipeline_time requires a supported schedule, complete effective cost profile, and occupancy inputs`, appears on 1,481 candidates. It is a downstream symptom, not another independent cause. The other 28 return earlier from the demand gate. The detailed audit retains exact module messages instead of relying only on this generic summary.

The inter-warp reduction restriction did not remove a currently compilable attention candidate in this experiment. It remains a necessary modeling task once the layout path is supported. By contrast, all 28 demand-excluded candidates compiled and passed correctness. Their exclusion is a model policy decision, not a demonstrated compiler impossibility. Two larger noncausal attention oracle winners are excluded by both the stage and demand gates.

**P0: parallel scalar work is counted once instead of once per tile element.** In [collector.py](../tilelang/tiletune/src/collector.py), `str(node.kind)` stores parallel loops as `"1"`. In [compute.py](../tilelang/tiletune/compute.py), `operation_work` compares that value with `str(tir.ForKind.PARALLEL)`, which is `"ForKind.PARALLEL"` in this checkout. The filtered dimension list is empty and its product becomes one.

The minimal 64×64 `T.exp2` reproduction reports `exp_ops=1`; it must contain 4,096 logical exponential evaluations. The saved attention score-tile phase has the same error. Row-kernel and fused-GEMM arithmetic is also affected. There are 636 candidate-case pairs with nontrivial parallel scalar domains across 18 held-out cases. This is a shared implementation bug affecting all backends, independent of GPU rates.

Repair requires a consistent loop-kind representation and exact work-count regression tests for one- and multidimensional tiles, row broadcasts, symbolic extents and loop-contained scalar phases. The existing attention tests mostly assert that exponential work is greater than zero; that does not detect a factor-of-4,096 error. The analysis version/cache identity must change when the arithmetic changes.

The exploratory replay corrects only this multiplier. Attention's median Spearman correlation changes from −0.217 to approximately +0.157, but its six Oracle@K values remain unchanged. Its median predicted/measured ratio increases from 1.759 to approximately 2.365: correcting an undercount can expose overestimates elsewhere. This demonstrates error cancellation; it does not validate the corrected model or justify retaining a known bug. No production source was patched and no new shortlist was benchmarked.

**P0: positive-stage timing is tied to one Hopper policy.** [pipeline.py](../tilelang/tiletune/pipeline.py) requires `warp_specialization.status == "predicted"` whenever stages are positive. [warp_specialization.py](../tilelang/tiletune/warp_specialization.py) predicts only `sm_90`/`sm_90a`, automatic layouts, one recognized pipeline and supported pure-TMA producers. It rejects or leaves unknown manual warp specialization, mixed/non-TMA copies, conditional producers, multiple pipelines, explicit layouts and unsupported thread partitions.

Ampere stage 2/3 is consequently unavailable despite working compiler/runtime support. The same gate excludes positive-stage timing on Blackwell and HIP. Even Hopper's ordinary non-specialized software pipelines fail this gate. The current max-plus recurrence is useful infrastructure for buffer readiness, reuse and startup, but its eligibility and issue/synchronization policy must be separated from Hopper warp specialization.

Implement distinct scheduling policies for synchronous copies, ordinary asynchronous-copy software pipelines, specialized TMA pipelines, and backend-specific asynchronous matrix operations. Derive commit/wait/barrier groups and buffer lifetimes from the compiler-selected path. Ampere provides asynchronous global-to-shared copies and split arrive/wait barriers; these are real scheduling mechanisms that require representation in the model. [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/archive/13.0.2/ampere-tuning-guide/index.html).

**P0: a recognized operation can silently contribute zero work.** `operation_work` initializes every component to zero and has no final unsupported-operation branch. Native parsing/access-region support is therefore sufficient for some unmodeled operations to pass through the timing pipeline.

The offline reproductions establish three concrete cases:

| Operation | Reproduction result | Defect |
| --- | --- | --- |
| Shared-memory transpose | Finite score, empty unknown list; transpose work all zero | No transpose instruction, transaction, shared-bank or synchronization cost |
| Tile atomic-add | Finite score, empty unknown list; atomic operation work all zero | No atomic read-modify-write service or contention model; logical traffic alone is insufficient |
| Cumsum | Unknown reduction with `'CumSumOp' object has no attribute 'type'` | Attribute-based dispatch confuses scans with reductions; `src-dst` element counting also gives zero for an equal-size scan |

These operations were not in the measured portable suite, so no runtime-performance number is claimed for them. They show that “finite score” does not currently guarantee that every operation has a model. Use explicit operator/instruction contracts: an unimplemented operation must return a specific unsupported reason, never zero cost by default. Scans need their own work/dependency model; transpose and atomics need their own service and communication models.

**P1: the occupancy proxy omits much of the compiler's register allocation.** [occupancy.py](../tilelang/tiletune/occupancy.py) uses the larger of accumulator storage and live logical tile storage outside the recognized Hopper policy. Compiler operand fragments, address calculations, temporaries, layout replication, allocation granularity and scratch are absent. The proxy is not an upper bound on actual allocation.

Across the 220 compiled timing selections, the median compiler register count is **2.156×** the proxy per thread. Merely substituting compiler registers in the same resource formula reduces the resident-CTA bound for **210/220** records. These comparisons concern the compiled shortlist, not the complete grid; the revised bound still ignores allocation granularity and is not a measurement of achieved occupancy.

For `flashattention.test1`, candidate 0 predicts 81.5 registers/thread and six resident CTAs/SM; the compiler reports 150 registers/thread, permitting at most three by the same register-capacity calculation. Candidate 19 predicts three resident CTAs but the compiler count permits at most one. Wrong residency affects both the number of waves and the contention factor used for memory/compute service, so its effect on latency is not a single constant multiplier.

Conversely, conservative live intervals can overestimate demand and exclude valid configurations. The 28 demand exclusions and four zero-residency cases demonstrate this. Relaxing the gate alone would leave spill/reuse effects unresolved. Model instruction operands and layout ownership, keep hard feasibility proofs separate from estimated demand, expose resource uncertainty, and validate against compiler counters. None of the 220 compiled timing selections reports spill or local-memory bytes, so spills are not an observed explanation for their errors. Compiler data for every excluded candidate were not collected by the independent oracle.

**P1: logical traffic is treated as physical service at one memory rate.** [global_memory.py](../tilelang/tiletune/global_memory.py) explicitly excludes transaction/coalescing and inter-block cache modeling. Every logical external tile visit is charged; [pipeline.py](../tilelang/tiletune/pipeline.py) uses a single `global_bytes_per_cycle` for reads and writes. The experiment selected the streaming profile globally. The cached rate was measured but no working-set or reuse model selects or mixes rates per access.

For GEMM NN 2048³, candidate 0 predicts 0.923 ms versus 0.319 ms measured. Logical repeated input loads total 1 GiB although unique inputs occupy 16 MiB and the A100 reports 40 MiB L2. Copy service is 89% of its modeled iteration. Streaming versus cached service arithmetic changes the input component from approximately 0.644 ms to 0.284 ms. This supports investigating cache/repeated-load accounting; cache misses, physical traffic and causality were not measured. Kernel-invocation cache flushing does not eliminate reuse between CTAs during the kernel.

The model also lacks sector/transaction amplification for strided or misaligned loads, separate read/write service, cache topology/partition effects, bank conflicts, shared-layout swizzles and matrix operand load instructions. Shared bytes are explicitly counted for GEMM operands, but ordinary shared-memory scalar accesses and shared copies do not have a general physical-traffic accounting path. These are relevant across backends; transposed, masked, scattered and fused operations make them particularly visible. A single fitted global rate cannot distinguish them.

**P1: primitive profiles do not isolate all relevant costs.** [device_profile.py](../tilelang/tiletune/profiling/device_profile.py) measures fixed, reusable kernels, which avoids candidate-label fitting. However:

| Profile limitation | Evidence or consequence |
| --- | --- |
| Copy residual can become physically uninformative | Clean A100 copy roundtrip is 16.342 cycles; subtracting 4,096/24.801 bytes-per-cycle and the 28.951-cycle barrier gives a negative residual, which is clamped to `copy_latency_cycles=0`. Zero here is a failed separation of effective probe costs, not evidence of zero hardware latency. |
| Copy path mismatch | Ampere profiling measures a synchronous copy/dependent-consumer path. It provides no dedicated `cp.async` issue, commit/wait-group or overlap measurements. Hopper's probe measures TMA but other copy paths remain distinct. |
| Scalar instruction classes collapsed | A fixed FP32 FMA probe supplies service for add, multiply, division, casts, selects and other scalar expressions. Exponential probes and eight independent chains have their own dependencies/overheads; real kernels may have fewer ready chains or different dtypes. |
| One matrix tile and aggregate rate | The matrix probe uses 64×128×128 and 256 threads for aggregate service. Only WGMMA has a separate per-warpgroup ceiling. MMA/MFMA/WMMA tile shape, ready warps and operand dependencies can limit throughput differently. |
| Thread-domain holes | Consumer rates exist only for 32, 64, 128, 256 and 512 threads. A 384-thread Hopper partition is recognized by the policy but lacks a scalar/reduction consumer-rate row; such phases can yield unknown timing. Other legal counts also need explicit handling. |
| Dtype/instruction holes | Automatic probes accept FP16/BF16/CUDA FP8 inputs and FP32 accumulation. There is no automatic coverage for FP32/TF32/FP64, integer matrix operations, mixed A/B dtypes, block-scaled FP4/FP8, sparse matrix instructions, HIP or other backends. |
| Environment and uncertainty | One reference SM clock and median primitive samples do not establish stability under different power states, memory clocks, partitions or contention. No uncertainty interval is propagated into the score. Cached-profile identity checks exist, but direct offline loading can omit expected-identity validation. |

Use instruction-specific, latency-sensitive and throughput-sensitive probes with source/instruction verification and positive, identifiable residuals. Store thread count, dtype, copy mechanism, issue/dependency pattern and memory regime with each measurement. Reject an unusable decomposition or expose its uncertainty. A workload latency anchor changes only a global scale and cannot repair missing coverage or ranking; none was used here.

**P1/P2: the schedule and shared-resource model remain approximations.** [compute.py](../tilelang/tiletune/compute.py), [schedule.py](../tilelang/tiletune/schedule.py) and [shared_memory.py](../tilelang/tiletune/shared_memory.py) already model useful quantities: dense GEMM work, mapped local/shuffle reduction pairs, buffer reuse, startup, stage storage and one-axis nonuniform CTA work. Missing or restricted cases include:

| Area | Current boundary | Required extension |
| --- | --- | --- |
| Reduction ownership | Explicit full-fragment layout or a direct CUDA MMA/WGMMA producer; sum/max only; one subgroup butterfly | Generic fragment layout propagation, inter-warp/block reductions, partial regions, other operators/dtypes, scans, native collectives, MFMA/WMMA/TCGEN05 layouts |
| Scalar math | Narrow intrinsic allowlist; RMSNorm `rsqrt` unavailable; enum bug undercounts supported expressions | Explicit work for reciprocal, sqrt/rsqrt, log, tanh, sigmoid, comparisons, integer/address operations, conversions and dtype-dependent lowering |
| Recurrence | Direct scalar global reads inside the recognized loop are rejected | Per-iteration global accesses, loop-carried register/shared state, actual dependencies and overlap |
| General control flow | One recognized main loop; nested serial loops and predicates are unsupported | Multiple/nested pipelines, branch/tail masks, data-dependent/gather accesses, symbolic bounds and safe uncertainty |
| Producer buffers | One distinct write per shared destination; known positive extent; first/last consumer | Region-aware partial writes, multiple writers, aliases, conditional copies, asynchronous stores and reuse across loops |
| Issue and dependencies | Serial sum of consumer phases; overlap recurrence mainly describes producer buffers; shared service uses fixed aggregate ceilings | Actual instruction DAG/critical paths, async matrix completion, resource-specific issue, synchronization groups and latency hiding |
| Shared allocation | Logical stage multiplication and conservative lifetime arena | Compiler padding/alignment, barriers and scratch, exact versioning/reuse policy, bank/swizzle costs; separate tensor memory where applicable |
| CTA/grid scheduling | Equal resident slots; fixed contention; uniform waves or one varying block axis up to 4,096 points; bounded dispatch enumeration | Partial-wave/dynamic-concurrency effects, multi-axis irregular work, persistent kernels, grouped/ragged workloads, clusters and multi-CTA cooperation |
| Small kernels and exits | Outside-loop work is summed; no distinct launch/dispatch or general asynchronous drain term | Separate startup/epilogue/drain, phase transitions and measured launch floors where they affect ranking/calibration |
| Spill allowance | Conditional timing explicitly omits spill traffic | Compiler-reuse-aware demand and backend-specific scratch/spill service; preserve uncertainty before compilation |

For an example already tested, causal attention's varying loop count is collected, so it would be wrong to claim that causal work is entirely unmodeled. What remains approximate is dispatch and shared-SM contention. Similarly, stage-dependent shared-memory allocation is already estimated even though Ampere positive-stage execution timing is missing.

**P1: unknown scores become a systematic search blind spot.** [ranking.py](../tilelang/tiletune/ranking.py) selects only finite eligible scores, retains original grid order on ties, and excludes unknown/demand-gated candidates even in report-only mode. There is no exploration allocation to unsupported but compilable candidates. The timing search therefore never measures the best schedules in this experiment.

The traffic score, `(bytes_per_block + 1) × waves`, supplies broader coverage but excludes compute, dependence and overlap quality. Different schedules can tie, with original grid order breaking the tie. For GEMM NN test1 it yields only 21 distinct finite scores across 108 configurations. Its GEMM Oracle@K GM is 0.697. It should remain an explicitly labeled heuristic with its own validation, not be presented as a complete timing fallback.

An engineering fallback can allocate a declared share of K to unknown candidates, diverse schedule families or a backend heuristic, while reporting analytical coverage separately. Choose that policy on development data before the next test. The existing no-replacement rule after compilation failure preserves this experiment's fixed budget; it can waste budget in production but did not cause timing's A100 selection loss: every finite timing candidate compiled successfully here. Traffic and XGBoost did select some compiler-invalid attention configurations. A shared compiler-feasibility summary and consistent budget accounting are needed for comparisons.

**GPU coverage: architecture names are insufficient.** [targets.py](../tilelang/tiletune/targets.py) supplies target identity and coarse semantics; it does not implement all of the hardware paths implied by those names. Model dispatch should use the selected instruction, copy mechanism, subgroup/partition, storage scopes and synchronization policy, plus measured capacities. The following is the current code boundary, not a claim that every named device has been tested.

| GPU/backend | What exists | Missing coverage needed |
| --- | --- | --- |
| NVIDIA Ampere `sm_80/86/87` | Target/resource detection, automatic MMA profiles for supported dtypes, restricted serial timing | Ordinary async-copy stage scheduling; operand/temporary registers, MMA issue/dependency ceilings; generic reductions and common defects above. Only A100 `sm_80` was measured. |
| NVIDIA Ada `sm_89` | CUDA identity/register cap and compiler paths | Target family resolves to generic `cuda`; automatic primitive profiling rejects it. Add measured MMA/FP8 capabilities and applicable copy/schedule policies, not A100 capacity constants. |
| NVIDIA Hopper `sm_90/90a` | WGMMA profiles and narrow automatic pure-TMA warp-specialized recurrence | Ordinary pipelines, mixed copies, explicit/manual layouts and partitions, missing consumer domains, asynchronous WGMMA groups, TMA stores/multicast and cluster execution |
| NVIDIA Blackwell paths with TCGEN05/TMEM | Family recognized; automatic profiles currently choose legacy `cuda.mma` | Native `cuda.tcgen05` signature/rates, tensor-memory capacity/liveness/service, completion barriers, 2CTA/cluster schedules, scaled operands; current positive-stage gate excludes it |
| NVIDIA `sm_120/121` | Recognized as Blackwell, legacy MMA profile path; compiler includes an SM120 block-scaled MMA implementation | Separate actual instruction/capability contract and profiles for `cuda.mma.blockscaled`; applicable asynchronous-copy schedules. Do not apply SM100 TMEM/TCGEN05 assumptions solely from the Blackwell family name. |
| Earlier NVIDIA architectures | Several register caps and CUDA compiler paths are recognized | Automatic profiling rejects them; native FMA/MMA selection, older copy/synchronization behavior, measured profiles and reductions need explicit coverage |
| AMD CDNA/GCN `gfx9*`, including `gfx942` and `gfx950` | HIP identity, subgroup 64, native worker protocol, external matching MFMA profiles; some compiler counters | No automatic HIP primitive probes; no positive-stage policy; no automatic MFMA reduction layout; VGPR/AGPR/SGPR and per-SIMD residency, scratch, LDS, wait counters and cache topology require backend models |
| AMD RDNA `gfx10/11/12*` | HIP normalization defaults to subgroup 32; native ROCm WMMA path exists | Actual wave mode/instruction-specific model, RDNA register/LDS allocation and scheduling, WMMA profiles/layouts and appropriate dtype capabilities |
| Metal, Intel/other GPU backends | No native TileTune target/timing model in this resolver; portable manifest does not admit these backends | Backend integration, worker, subgroup/storage/resource model, instruction introspection and probes before any coverage claim |
| Ascend, an accelerator rather than a GPU | Logical analysis and external-worker extension point | No native core/storage timing or residency model; external execution alone does not fill this gap |

Hopper TMA and thread-block clusters introduce transfer and cooperation mechanisms beyond a per-CTA byte rate. [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/13.0.0/hopper-tuning-guide/index.html). Blackwell variants have different occupancy/storage limits, and TCGEN05's tensor-memory operations require their own execution contract. [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), [PTX TCGEN05 specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions). AMD documents distinct register files, wavefront capabilities, LDS and cache configurations across CDNA and RDNA; those distinctions cannot be represented by changing only SM count and bandwidth. [AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).

The portable experiment support predicate also needs capability-based dtype checks: its HIP FP8 branch currently assumes MI308-style FNUZ requirements for every HIP target. That is a runner coverage defect independent of timing. Validate formats against the actual GPU and selected matrix instruction. Backend acceptance, dtype support, compiler availability, model coverage and runtime validation should be reported as separate statuses.

**Implementation order and acceptance criteria.** These are proposed requirements for follow-up work, not achieved results or fitted thresholds.

| Priority | Concrete work | Evidence required before calling it complete |
| --- | --- | --- |
| P0 | Fix loop-kind work counting; explicit unsupported-operation dispatch; typed scan/reduction handling | Exact IR work counts, no silent zero-cost nontrivial operators, minimal reproductions converted into regression tests |
| P0 | Separate schedule eligibility from Hopper warp specialization; add ordinary Ampere pipelines | Stage 0/1/2/3+ tests against compiler-selected copies/waits; finite scores for the currently valid GEMM/attention/chunk grids; stage winners remain eligible |
| P1 | Generic reductions, RMSNorm math, recurrence accesses | Complete timing coverage on all currently valid row/recurrent adapters; correct ownership/communication counts and explicit reasons for genuinely unsupported paths |
| P1 | Resource/occupancy and copy-profile repair | Predicted versus compiler register/shared allocations; legal occupancy bounds; interpretable copy latency and commit/wait measurements; separately report uncertainty and scratch |
| P1 | Physical traffic/cache and instruction service | Counter-supported byte/cache/bank/issue diagnosis, microbenchmarks that vary one mechanism, and calibration by phase, tile, thread count and dtype |
| P1 | Capability-based backend policies and profiles | Per-instruction backend tests for Ampere, Ada, Hopper, both relevant Blackwell instruction families, CDNA and RDNA; explicit unsupported states for absent backends |
| P2 | Clusters, persistent/ragged/irregular kernels, atomics/scans/sparse/quantized operators | Dedicated adapters, references, capability checks and matched exhaustive comparisons for each new feature |
| P2 | Unknown-candidate exploration and compiler feasibility | Predetermined budget policy, equal failure accounting, top-1/top-K curves, coverage ceiling, and random-search comparison |

A common analysis graph should describe operations, regions, ownership, loop dependencies and buffers. Backend policies should supply storage capacities/allocation rules, copy and matrix instruction semantics, synchronization, issue limits and measured primitive profiles. A structured capability result should identify the unsupported operation or schedule feature. This lets one runner exercise different GPUs without pretending their schedules are interchangeable.

**Validation and generalization boundaries.** Keep the current test set as a regression/debugging set; it has now informed defect diagnosis. Use untouched final cases for future quality claims: new aspect ratios, irregular dimensions/alignment, sequence/head dimensions, dtypes, layouts, masks, stage counts, thread counts and algorithm implementations. Include each supported instruction/copy family on actual hardware; add lower-level cross-compilation checks when hardware is unavailable, while keeping them distinct from runtime validation.

For every GPU and workload, report hardware/compiler support, finite score coverage among compilable candidates, oracle-winner coverage, coverage ceiling, top-1/top-K retained performance, rank correlation, absolute/relative error, resource-estimation error, compilation failures, tuning cost and uncertainty. Report small grids separately; near-oracle performance with K=6 of eight scored choices gives weak evidence about ranking. Compare baselines only on matched cases and budgets. Carver's current adapter covers 12 plain-GEMM tests, not the full 34-case suite.

TileTune used no workload-label fitting or latency anchor in this experiment, and the audit found no frozen-source mismatch. That prevents one form of leakage; it does not establish absence of overfitting or broad portability. The XGBoost baseline's strong GEMM shortlist quality coexists with poor absolute extrapolation: GEMM test log-RMSE 2.898 versus training 0.043. Its 97.2% matched-GEMM Oracle@K is not evidence that its latency predictions generalize across shapes or GPUs.

The clean run observed no unrelated compute process at 282 measurement boundaries, but did not continuously enforce isolation. Winner remeasurement spread had median 2.60% and maximum 17.01%; small differences and the near-oracle row/chunk results require caution. The initial contended run is excluded. No GPU counters were collected for this audit, so cache/issue/bank hypotheses remain unisolated. Claims of covering all GPUs require the backend work and runtime matrix above; neither passing the existing tests nor loading an architecture-tagged profile establishes that coverage.

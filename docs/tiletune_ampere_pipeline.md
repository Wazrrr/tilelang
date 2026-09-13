**Ampere pipeline scoring and validation**

2026-09-14. Final analysis version 20, device-profile version 5. Prototype version 19 is retained as development evidence. The target is A100 80GB PCIe (`sm_80`). The ranking metric remains `pipeline_time`: its score is estimated grid cycles, and lower is better. This work follows the defects documented in the [version-18 audit](tiletune_model_audit.md).

The [Ampere model](../tilelang/tiletune/ampere.py) reads the compiler's software-pipeline stage, order and asynchronous-copy-group annotations from a separate IRModule. Ordinary Ampere warps issue copies and execute consumers in the compiler's order. The schedule carries asynchronous byte-service completion and per-iteration readiness through startup, steady state and drain. It also handles loops shorter than the pipeline. Positive stages no longer require a Hopper warp-specialization policy. Analysis does not compile or benchmark a candidate, or mutate the original PrimFunc.

The implementation also corrects parallel scalar work counts, adds reciprocal-square-root service, and prevents unsupported parsed operations from silently receiving zero compute work. Generic reductions use compiler-inferred fragment ownership; supported inter-warp XOR rounds include shared traffic, combination work and barriers. Recurrent global reads and stores are charged at their actual loop phase. The occupancy estimate includes native MMA operand fragments for operands loaded from shared memory.

The new asynchronous-copy probe measures issue time and readiness through `cp.async.wait_group` and a dependent shared load. Readiness is a measured minimum combined with byte-service completion using `max`; it is not a residual obtained by subtracting an aggregate bandwidth measurement. Profiles also contain independent `rsqrt` rates. Older profiles remain loadable, but positive-stage Ampere timing requires the new asynchronous-copy fields. No workload latency anchor or candidate-cost fitting is used. The fresh A100 probe measured 304.07 cycles of readiness and 230.51 issued bytes/cycle; the aggregate `rsqrt` rate was 15.66 operations/cycle/SM. These numbers describe the fixed probes, not a universal hardware latency or peak throughput.

**Validation protocol.** The final TileTune/experiment suite passed 395 tests, skipped 13 and deselected three Blackwell-specific tests on this Ampere host. The [test log](../experiments/results/ampere-pipeline-v20/test-results.txt) records the version-20 run. The [Ampere tests](../testing/python/tiletune/test_ampere.py) cover stages 0–3, unchanged input IR, generic reductions, recurrent external accesses, scalar work, startup/drain and exact agreement with explicit event replay. A 100-million-iteration synthetic case verifies that scheduling does not unroll the recurrence. The [full-planning cross-check](../experiments/results/ampere-pipeline-v19/native-plan-check.json) also found exact agreement for 124 positive-stage plans: 72 GEMM, 36 attention and 16 chunk-KDA configurations.

The final experiment uses the complete portable grids and the same declared comparison budget: `min(20, ceil(0.1 * grid_size))`. Training scales are 0.25 and 0.5; validation is 0.75. The fresh test scales are 1.25 and 1.5. The previously inspected scales 1 and 2 are development data for this revision. Their replay can measure regression and coverage, but cannot establish unseen-shape generalization. TileTune pipeline timing is the primary ranker; brute force, Carver and XGBoost remain comparison methods. Traffic ranking is retained as a separately declared diagnostic baseline.

```bash
CUDA_HOME=/root/cuda-12.4 \
CXX=/usr/bin/g++-10 \
PYTHON=/root/tilelang/.venv/bin/python \
bash experiments/portable/run_accelerator.sh \
  --device ampere --workers 8 --warmup 10 --rep 50 \
  --validation-repeats 7 --wait-idle \
  --test-scales 1.25 1.5 \
  --output experiments/results/ampere-pipeline-v20-clean
```

**Completed fresh A100 experiment.** All 38 declared test cases finished: 34 supported cases passed correctness, and four FP8 cases were explicitly unsupported. Exhaustive testing measured 1,998 correct configurations out of 2,160; the other 162 failed compilation. TileTune scored 1,964 configurations, all of which compiled correctly. Results, frozen rankings, compiler observations and measurements are in [the final experiment directory](../experiments/results/ampere-pipeline-v20).

TileTune is a useful shortlist ranker on these A100 cases, but its absolute latency model and first-choice precision remain incomplete. Across the 34 cases, its frozen shortlists retain **98.31%** of exhaustive-best performance, versus **95.00%** for XGBoost. Repeated measurements of the kernels actually chosen give **98.18%** and **95.11%** respectively. These are geometric means across cases. Giving each of the eight families equal weight instead gives TileTune 98.57% shortlist performance. The primary ranker is always `pipeline_time`; the diagnostic traffic ranker reaches 73.91% overall and is never substituted according to the oracle.

`Oracle@K` is exhaustive-best latency divided by the fastest oracle latency among the frozen shortlist. It measures the shortlist, without assuming its separately timed winner is always the same configuration. The exhaustive reference is 100%. The table uses geometric means within each family; Carver is unavailable outside its supported adapter domain.

| Family | Fresh cases | TileTune Oracle@K | XGBoost Oracle@K | TileTune chosen kernel, remeasured | TileTune tuning speedup vs exhaustive |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM | 16 | 98.27% | 96.52% | 98.09% | 6.38× |
| attention | 6 | 97.45% | 94.70% | 97.17% | 2.03× |
| recurrent KDA | 2 | 100.00% | 100.00% | 100.00% | 0.60× |
| chunk KDA | 2 | 96.37% | 99.34% | 95.96% | 3.02× |
| softmax | 2 | 97.79% | 66.51% | 97.82% | 0.91× |
| RMSNorm | 2 | 99.56% | 99.85% | 99.56% | 0.91× |
| reduce-sum | 2 | 99.74% | 100.00% | 100.15% | 1.29× |
| elementwise | 2 | 99.47% | 99.13% | 99.48% | 1.71× |

On the **same 12 Carver-supported GEMM cases**, shortlist performance is TileTune **98.01%**, XGBoost **96.01%**, and Carver **90.56%**. Carver does not support the portable batched or fused-epilogue variants, or the other families, so its 12-case average must not be compared directly with the other methods' 34-case averages. See the [per-case table](../experiments/results/ampere-pipeline-v20/analysis.tsv), [matched summaries](../experiments/results/ampere-pipeline-v20/final-tables.json), and [shortlist plot](../experiments/results/ampere-pipeline-v20/oracle-at-k.png).

The recorded tuning times exclude worker startup and shared preparation; kernel and autotuning caches are disabled by the coordinator. Across all test cases, exhaustive tuning took 1,829.5 s, TileTune 468.7 s, and XGBoost 333.4 s; the geometric mean of per-case speedups is 2.91× for TileTune and 4.03× for XGBoost. TileTune's primitive profiling took another 196.0 s. XGBoost's exhaustive training/validation collection and fitting took another 2,591.0 s. These preparation costs are amortized differently and are not free. On the six-entry recurrent-KDA, softmax and RMSNorm grids, analytical work costs enough that TileTune is slower than exhaustive tuning despite good selections.

Winner validation uses seven rounds in shuffled order, sharing measurements when methods choose the same configuration. Median relative measurement spread is 1.70% for TileTune winners and 2.16% for exhaustive winners. Remeasured ratios slightly above 100%, such as reduce-sum, reflect measurement variability and independently selected winners; they are not evidence of beating the exhaustive search space.

**Prediction quality and pipeline-stage behavior.** The following correlations and latency ratios use exactly the same successful candidates with finite predictions from both methods. Values are medians within each case and then across cases. GEMM and attention shortlist quality is stronger than first-choice performance: TileTune's top-ranked configuration retains only 83.60% and 86.87% of exhaustive performance respectively. Overall top-1 performance is 89.24%.

| Family | TileTune eligible / grid | TileTune rank correlation | XGBoost rank correlation, matched candidates | TileTune predicted / measured latency | XGBoost predicted / measured latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM | 1728 / 1728 | 0.844 | 0.827 | 2.801× | 0.069× |
| attention | 144 / 324 | 0.646 | 0.799 | 1.631× | 0.257× |
| recurrent KDA | 12 / 12 | 1.000 | 1.000 | 0.725× | 0.371× |
| chunk KDA | 48 / 48 | 0.241 | 0.268 | 0.541× | 0.675× |
| softmax | 6 / 12 | 0.750 | −0.750 | 1.111× | 0.169× |
| RMSNorm | 6 / 12 | 0.250 | −0.250 | 1.127× | 0.204× |
| reduce-sum | 10 / 12 | 0.900 | 0.900 | 0.879× | 0.257× |
| elementwise | 10 / 12 | undefined | −0.550 | 1.065× | 0.202× |

Elementwise has tied TileTune scores, so correlation is undefined; its good selected performance does not establish useful ordering. RMSNorm has only three eligible entries per case, and weak median ordering. Chunk KDA is the clearest fresh ranking failure: on its first shape, correlation is −0.039 and Oracle@3 is 95.11%, below the saved random-shortlist mean of 97.29%. Near-best performance on a relatively flat grid does not demonstrate a sound cost model. See [matched prediction errors](../experiments/results/ampere-pipeline-v20/prediction-error-comparison.json) and the [latency scatter plot](../experiments/results/ampere-pipeline-v20/prediction-scatter.png).

The predicted/measured ratio describes bias, not absolute error. Opposite errors can cancel when summarizing across shapes: softmax's ratio is 1.111×, but its median absolute relative error is 35.6%, with per-case median ratios of 0.813× and 1.409×. The linked error table preserves both measures.

Positive stages are now scored using their actual compiler plans, but modeling support is not the same as accurate stage preference. The [controlled stage comparison](../experiments/results/ampere-pipeline-v20/stage-comparison.json) pairs configurations that differ only in stage count:

| Family | Stage-0 / positive-stage pairs | Median predicted speedup | Median measured speedup | Correct direction when measured change exceeds 10% |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 1152 | 1.279× | 1.273× | 990 / 1040 |
| attention | 96 | 1.474× | 1.276× | 87 / 87 |
| chunk KDA | 32 | 1.069× | 0.962× | 1 / 9 |

Chunk KDA demonstrates a remaining stage-preference defect: the model generally predicts a benefit where measurement shows a small slowdown. Attention overestimates the magnitude of pipelining gains. The 10% threshold is descriptive, not a statistical confidence bound. Instruction-level issue, synchronization, register allocation, and short-loop overhead remain approximations; this experiment does not causally apportion the stage-prediction error among them.

**Isolation and overfitting checks.** The [integrity audit](../experiments/results/ampere-pipeline-v20/audit.json) found no model/experiment source changes, and all 38 ranking snapshots match their recorded hashes and precede exhaustive requests. All 38 XGBoost requests match models written before those requests. The [additional provenance audit](../experiments/results/ampere-pipeline-v20/provenance-extra-audit.json) confirms disjoint training, validation and fresh-test shapes, no fresh test shape in the earlier experiment, unchanged compiler helper sources, and unscaled version-20 `pipeline_time` predictions. None of the 257 current-run GPU-boundary observations contains another compute process. The earlier reused training observations contain only that run's coordinator.

TileTune has no candidate-latency fit in this experiment. That prevents this form of label leakage; it does **not** prove absence of development overfitting or generalization to other A100 shapes, Ampere SKUs, or kernel implementations. The model was developed using the old 1×/2× cases, and these fresh 1.25×/1.5× tests cover the same declared families on one A100. No analytical model changes were made after examining fresh test results.

The frozen XGBoost models show a substantial fit/generalization gap. GEMM log-latency RMSE is 0.048 on training, 0.923 on validation and 2.563 on fresh tests; attention is 0.014, 0.460 and 1.448. All recomputed validation errors match the saved training artifacts. Because the held-out shapes are larger than the training shapes, this combines size extrapolation and generalization failure; the experiment cannot attribute the entire gap to overfitting alone. See the [post-hoc frozen-model fit audit](../experiments/results/ampere-pipeline-v20/xgboost-fit-audit.json). These are the declared baselines with fixed settings, not a claim that all possible XGBoost models would perform this way.

The version-19 collection stopped before any fresh test selection after offline assembly inspection found synchronization and scalar-ownership defects. Version 20 corrects them. Its [reuse record](../experiments/results/ampere-pipeline-v20/reused-training-provenance.json) preserves 25 completed brute-force training cases, with original file hashes; all hashes still match, and brute force bypasses the analytical model. Kernel, compiler and benchmark settings are unchanged. The recorded run used its saved manifest and `--resume` to reuse the verified training prefix. The command above reproduces the complete protocol from scratch in a separate directory.

**Version-20 development replay (previously inspected shapes).** Re-scoring the original 34 supported cases increases eligibility from 651 to 1,970 of 2,160 candidates, covering 98.6% of the 1,998 correct oracle candidates. This is [replay evidence](../experiments/results/ampere-pipeline-v20/development-summary.json), using saved oracle timings and fresh primitive profiles, without recompiling or timing candidates. It is not an independent test of generalization. The superseded [version-19 prototype replay](../experiments/results/ampere-pipeline-v19/development-summary.json) is retained separately.

| Family | Eligible v18 → v20 | Oracle@K v18 → v20 (geometric mean) | v20 median rank correlation | v20 median predicted/measured latency |
| --- | ---: | ---: | ---: | ---: |
| attention | 48 → 144 / 324 | 0.732 → 0.950 | 0.763 | 1.763× |
| elementwise | 11 → 11 / 12 | 0.994 → 0.994 | 0.655 | 1.088× |
| gemm | 576 → 1728 / 1728 | 0.818 → 0.970 | 0.892 | 2.662× |
| kda_chunk_o | 16 → 48 / 48 | 0.995 → 0.976 | 0.070 | 0.673× |
| kda_recurrent | 0 → 12 / 12 | unavailable → 1.000 | 0.971 | 0.723× |
| reduce_sum | 0 → 11 / 12 | unavailable → 1.000 | 0.657 | 0.881× |
| rmsnorm | 0 → 8 / 12 | unavailable → 0.996 | 0.150 | 1.127× |
| softmax | 0 → 8 / 12 | unavailable → 0.959 | 0.600 | 1.231× |

On the same 220 previously compiled candidates, adding MMA operand fragments reduces the median compiler-register/model-proxy ratio from 2.156× to 1.659×. The [matched register comparison](../experiments/results/ampere-pipeline-v19/development-register-comparison.json) shows improvement in the storage estimate, while substantial compiler scratch, address state and allocation effects remain.

The fresh `gemm_nn.test1.25` report gives a concrete remaining occupancy error. First-ranked candidate 99 uses a 128×128×32 tile, stage 2 and 256 threads. The model estimates 22,528 registers per CTA and two resident CTAs. Compilation reports 131 registers per thread: even ignoring allocation granularity, two CTAs need 67,072 registers and exceed the SM's 65,536-register capacity. Candidate 104, ranked third, uses 128 threads and 224 compiler registers per thread, allowing two CTAs within the register limit. Their exhaustive timings are 0.255 ms and 0.174 ms respectively. The [saved report](../experiments/results/ampere-pipeline-v20/ampere/test/gemm_nn.test1.25/tiletune/tiletune.json) demonstrates a real occupancy-bound error and a misranking; it does not isolate how much of the latency difference occupancy causes.

The larger eligible set explains much of the recovered performance. GEMM and attention ordering is useful on these development cases, while chunk KDA and RMSNorm still have weak ordering. High Oracle@K on six-entry grids must be interpreted alongside rank correlation. Absolute GEMM timing remains substantially overpredicted. No model changes were made in response to this replay.

**Remaining boundaries.** These changes do not make every Ampere schedule predictable. The model still has one principal loop, unconditional operation mapping, one producer write per shared buffer, and bounded fragment ownership. Mixed GEMM/reduction graphs retain the existing MMA layout helper; unsupported attention layouts remain unscored. Generic reductions inside software pipelines, scans, transpose and atomics do not have complete timing implementations.

The [full model audit](../experiments/results/ampere-pipeline-v20/model-audit.json) identifies **34 valid candidates excluded by the conservative live-register-demand gate**, including five exhaustive winners: four non-causal attention cases and the first RMSNorm case. Coverage is 98.30% of correct candidates, with 29 of 34 exhaustive winners eligible. The 162 configurations with unresolved mixed-MMA reductions all fail compilation in this grid; they are a different category from the valid candidates rejected by the scoring gate.

In fresh `flashattention.test1.25`, 27 of 54 grid entries compile and pass correctness, while 24 receive finite TileTune scores. The valid but unscored configurations all use a 128×128 tile and 128 threads, at stages 0, 2 and 3. Their pipeline analysis is complete. The gate estimates 259 live registers per thread against a 255-register limit, although the proven accumulator lower bound is only 128; it does not establish physical overflow. The stage-2 configuration is the exhaustive winner. Even the best possible ranking of the eligible set reaches only 96.5% of that case's exhaustive performance. RMSNorm's excluded winner similarly exceeds the estimated limit by just two registers, at 257. The [case diagnostics](../experiments/results/ampere-pipeline-v20/ampere/test/flashattention.test1.25/comparison.json) retain the unscored winner explicitly. Fixing storage reuse/ownership and the gate's treatment of uncertain excess is a priority; simply widening the gate without modeling allocation or spills would hide the uncertainty.

Physical allocation is also underestimated for many eligible candidates. Among the 228 selected kernels with compiler resources, the median compiler/model register ratio is 1.523×. Replacing only the register bound with compiler counts reduces the resident-CTA upper bound below the model's estimate in 135 cases. Seventeen selected kernels report spill loads/stores and local storage whose traffic is absent from analytical timing. These selected-candidate observations are not an unbiased sample of the entire grid, and the substitution still omits allocation granularity. They establish that MMA operand fragments alone do not complete the register or spill model.

Cache reuse, transaction amplification, scalar instruction mix, exact compiler register allocation and storage reuse remain approximations or omissions. The conservative live-demand gate can still exclude compilable configurations. The asynchronous-copy rate is an effective fixed-probe rate, with vectorization and concurrency assumptions, not an instruction-level simulator. Finite scores and improved coverage alone do not demonstrate accurate ranking or absolute latency prediction. Runtime conclusions apply to the measured A100; other Ampere devices require their own profiles and validation.

The declared experiment uses the `streaming` memory regime throughout. It has no automatic cache-residency or cross-CTA reuse prediction. For the fresh 2,560³ GEMM, the two distinct FP16 input matrices occupy 26.2 MB, while the 32×32×32 configuration counts 2.10 GB of logical input tile traffic across CTAs. The A100 reports 41.9 MB of L2. In the saved model, memory service accounts for 88.8% of that configuration's loop interval. Streaming input service alone is estimated at 1.26 ms, compared with 0.610 ms measured for the entire kernel; the separate cached-rate calculation gives 0.557 ms of input service. These [read-only calculations](../experiments/results/ampere-pipeline-v20/detail-diagnostics.json) make the regime/traffic approximation a concrete suspect. They do not establish actual cache hit rates, a complete latency decomposition, or that switching the whole model to the cached rate would fix ranking. The frozen experiment does not select a regime from oracle timings.

Two report fields also need care when inspecting version 20. The Ampere `input_ready_latency_cycles` summary retains the legacy byte-service-plus-synchronous-residual expression; the positive-stage recurrence itself uses the independent asynchronous readiness probe. The detailed profile and schedule are authoritative for that path. Some inherited pressure assumptions still say operand fragments are unmodeled, although `ampere_mma_operand_registers` is included in the occupancy estimate. These are reporting inconsistencies, separate from the remaining timing and allocation approximations.


**Scalar and synchronization defects found during implementation.** For the development RMSNorm tile with 4,096 columns, one row and 128 threads, the prototype version-19 model counted 4,096 `rsqrt` evaluations. Generated CUDA places the same row-dependent expression inside a 32-element per-thread loop; the cubin contains one `MUFU.RSQ` site and no compute loop backedge. The compiler hoists the repeated expression, so that path executes 128 lane evaluations per CTA, not 4,096. This component was overcounted by 32× even after fixing the original parallel-domain bug. Version 20 uses compiler-inferred ownership and expression dependencies to count 128 evaluations, allowing reuse within each thread. See the [generated CUDA](../experiments/results/ampere-pipeline-v19/scalar-assembly/rmsnorm.cu), [disassembly](../experiments/results/ampere-pipeline-v19/scalar-assembly/rmsnorm.sass), and [offline compilation commands](../experiments/results/ampere-pipeline-v19/scalar-assembly/summary.json).

Recurrent KDA provides a related example: its compiler emits two exponential sites per thread in the recurrence, reusing results across locally owned value columns. Version 20 also corrects that count and accounts for replicated scalar outputs. Its scalar path excludes buffer-address arithmetic from the FP32 operation count. Separately, its direct dependent global loads are charged for byte service using the old zero synchronous-copy residual; the current model does not represent their load-to-use dependency latency. This is a concrete omission, but the fraction of the measured timing error caused by it has not been isolated. A separate scalar-load latency path remains necessary. The original version-19 replay is unchanged.

The same recurrent KDA kernel issues four scalar AllReduce calls per reduction per thread. Version 19 incorrectly shared their synchronization cost. Version 20 counts 16 inter-warp barriers and four workspace-reuse fences for each such reduction. Compiler barrier elimination can still reduce this conservative estimate. The correction follows the generated code and does not fit a coefficient to candidate timings.

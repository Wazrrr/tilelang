# TileTune evaluation on NVIDIA A100

2026-09-13. Checkout `dev`, base commit `113995b0afc8da1adaad009b1b896715b4d17176`, with the runner and CUDA-property compatibility changes described below.

The subsequent [Ampere pipeline revision](tiletune_ampere_pipeline.md) adds version-20 scoring for positive stages and generic reductions. The tables in this report retain the original version-18 measurements.

**TileTune's Ampere timing model has incomplete schedule coverage.** It produced a correct selected kernel for 26 of 34 hardware-supported held-out cases. The missing cases are recurrent KDA, softmax, RMSNorm and reduce-sum at both test sizes. The measurements below evaluate the current model without fitting its equations to these workloads.

The subsequent [model and GPU coverage audit](tiletune_model_audit.md) identifies additional defects in these same results: parallel scalar work is undercounted, compiler register usage exceeds the occupancy proxy, and conservative demand gates exclude compilable candidates. It inventories every observed exclusion reason and the missing architecture policies. The measurements and frozen selections below are unchanged.

The primary evidence is the [complete run](../experiments/results/ampere-full-comparison-idle/comparison.json), [derived table](../experiments/results/ampere-full-comparison-idle/analysis.tsv), [integrity audit](../experiments/results/ampere-full-comparison-idle/audit.json), and [GPU observations](../experiments/results/ampere-full-comparison-idle/gpu_observations.jsonl). The final run used fresh primitive profiles and training data, with `--wait-idle`. No unrelated compute process appeared in 282 recorded boundaries. The initial boundary had no compute processes; the remaining boundaries showed only coordinator PID 37906's retained CUDA context. This checks measurement boundaries rather than continuously enforcing exclusive scheduling.

An earlier run, retained at [ampere-full-comparison-contended](../experiments/results/ampere-full-comparison-contended/comparison.json), had a changing load environment: unrelated PID 9667 was present during primitive profiling, all training and early validation; it disappeared by 10:51:34 China time, before held-out tests began at 10:57:05. Its scores are preliminary and excluded from the main tables. The clean repeat used identical splits, grids, analytical equations and XGBoost hyperparameters; only the GPU-load policy, fresh primitive measurements and fresh training measurements changed. No unrelated process was stopped by this work.

## Direct baseline comparison

For the 12 plain, unfused, nonbatched FP16/BF16 GEMM tests where every baseline applies, these are geometric means over the same cases:

| Method | Cases | Oracle@K | Remeasured performance / brute force | Online tuning speedup |
| --- | --- | --- | --- | --- |
| TileTune time | 12 | 0.809 | 0.814 | 7.43× |
| TileTune traffic | 12 | 0.697 | 0.716 | 7.37× |
| Carver | 12 | 0.827 | 0.844 | 7.42× |
| XGBoost | 12 | 0.972 | 0.977 | 6.42× |
| Brute force | 12 | 1.000 | 1.000 | 1.00× |

`Oracle@K = fastest correct latency in the exhaustive grid / fastest correct latency among the frozen shortlist`, using one independent oracle table for both terms. Higher is better; 1 means the shortlist contains an oracle-best candidate. The brute-force baseline is exhaustive over the declared grid, not a claim of globally optimal tiling or a comparison with cuBLAS. Remeasured performance uses the method's actually chosen winner and the exhaustive winner, each measured again seven times in shuffled order; identical configurations are deduplicated. Those ratios can exceed 1 because selection and remeasurement are separate noisy observations.

Across all 16 GEMM tests, including batched and fused cases, Oracle@K is 0.818 for timing, 0.697 for traffic, and 0.972 for XGBoost. Carver has no implementation here for batched/fused GEMM, attention or the generic row/KDA kernels, so those entries remain explicitly unsupported. Averages over Carver's 12 cases must not be compared with averages over another method's 34 cases.

| Method | Completed / 34 supported tests | Remaining cases |
| --- | --- | --- |
| TileTune time | 26/34 | 8 model_unavailable |
| TileTune traffic | 34/34 | None |
| Carver | 12/34 | 22 unsupported |
| XGBoost | 34/34 | None |
| Brute force | 34/34 | None |

![Frozen shortlist quality by workload](../experiments/results/ampere-full-comparison-idle/oracle-at-k.png)

## Workloads, budgets and correctness

The portable suite contains 19 workload variants; 17 execute on this A100. CUDA FP8 and AMD FNUZ FP8 GEMM are explicitly unsupported on Ampere and were recorded as such in every split. This covers the portable experiment adapters, not every example or backward kernel in the repository.

GEMM uses base M=N=K=2048, with NN, NT, TN, batch=4, bias+ReLU, BF16, tall (8192×512×2048), and wide (512×8192×2048) variants. Attention uses B=1, H=16, S=2048, D=64, with noncausal FP16, causal FP16, and noncausal BF16 variants. Recurrent KDA uses B=1, H=8, S=256, key/value dimensions 64; chunk-output KDA uses S=1024 and chunk size 64. Row kernels use 4096×4096. KDA chunk output is one forward stage, not end-to-end chunked KDA.

Training scales are 0.25 and 0.5, validation 0.75, and test 1 and 2. GEMM scales all M/N/K; row kernels scale both dimensions; attention/KDA scale sequence only. Dimensions round to multiples of 32 or the chunk size. Canonical whole-workload identities reject aliases and rounding collisions across splits. All declarations are in the [frozen plan](../experiments/results/ampere-full-comparison-idle/plan.json).

The candidate grids contain 108 GEMM, 54 attention, 24 chunk-KDA, and 6 recurrent/row configurations. The online budget is `min(20, ceil(0.1 * grid_size))`: respectively 11, 6, 3 and 1. All methods use the same case grid; shortlist methods receive the same K, while brute force measures the full grid. Failed selected candidates consume budget and are not replaced. True brute force bypasses TileTune analysis, resource filters and prediction gates. TileTune timing is the primary metric; traffic scoring was declared separately before collection and is not a per-case fallback chosen from the oracle.

| Split | Completed cases | Unsupported FP8 cases | Correct candidate measurements | Compile failures |
| --- | --- | --- | --- | --- |
| train | 34 | 4 | 1998 | 162 |
| validation | 17 | 2 | 999 | 81 |
| test | 34 | 4 | 1998 | 162 |

All successful candidate timings passed the adapter's reference checks, including elementwise tolerances and a relative output-norm check. Compilation failures remain in each candidate table. Attention has 27 compilable candidates and 27 layout-inference failures per 54-candidate grid; this is a real compiler-feasibility limitation, not a numerical mismatch. The early small-shape traffic top-2 smoke failed all three attention variants because both selected layouts failed compilation; full-size grids still contain valid attention kernels.

Benchmarking uses `tvm_ffi`, CUDA events, L2-cache management from the standard profiler, seed 123, warmup target 10 ms, measurement target 50 ms, eight compilation workers, no timing-based early stopping, 30-second candidate timeout and 1800-second case timeout. Every case runs in a fresh process. Seven shuffled measurements validate each distinct final winner with the same reference contract. Across 140 method/case validation rows, the median `(max-min)/median` spread is 2.60%, the maximum is 17.01%, and 7 exceed 10%. These spreads are descriptive, not confidence intervals; small differences should not be overinterpreted.

## Every held-out result

`test1` means base size and `test2` the doubled dimensions described above. Time/Traffic/Carver/XGBoost columns are Oracle@K; dashes mean no applicable model or no measured shortlist. `Time scores` reports eligible finite scores out of the full grid, including candidates that might later fail compilation.

| Held-out case | Grid / K | Oracle ms | Time | Traffic | Carver | XGBoost | Time scores |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemm_nn.test1 | 108/11 | 0.0969 | 0.780 | 0.826 | 0.841 | 0.985 | 36/108 |
| gemm_nn.test2 | 108/11 | 0.6359 | 0.857 | 0.620 | 0.827 | 0.943 | 36/108 |
| gemm_nt.test1 | 108/11 | 0.0976 | 0.784 | 0.805 | 0.835 | 1.000 | 36/108 |
| gemm_nt.test2 | 108/11 | 0.6193 | 0.840 | 0.553 | 0.785 | 0.990 | 36/108 |
| gemm_tn.test1 | 108/11 | 0.0959 | 0.771 | 0.813 | 0.844 | 1.000 | 36/108 |
| gemm_tn.test2 | 108/11 | 0.6326 | 0.860 | 0.655 | 0.839 | 1.000 | 36/108 |
| gemm_batched.test1 | 108/11 | 0.3323 | 0.863 | 0.753 | — | 0.989 | 36/108 |
| gemm_batched.test2 | 108/11 | 2.7613 | 0.894 | 0.617 | — | 0.951 | 36/108 |
| gemm_bias_relu.test1 | 108/11 | 0.0987 | 0.783 | 0.827 | — | 0.987 | 36/108 |
| gemm_bias_relu.test2 | 108/11 | 0.6349 | 0.860 | 0.608 | — | 0.964 | 36/108 |
| gemm_bf16.test1 | 108/11 | 0.0960 | 0.773 | 0.823 | 0.837 | 0.988 | 36/108 |
| gemm_bf16.test2 | 108/11 | 0.6168 | 0.852 | 0.620 | 0.826 | 0.932 | 36/108 |
| gemm_tall.test1 | 108/11 | 0.0996 | 0.766 | 0.817 | 0.845 | 1.000 | 36/108 |
| gemm_tall.test2 | 108/11 | 0.6287 | 0.839 | 0.678 | 0.853 | 0.949 | 36/108 |
| gemm_wide.test1 | 108/11 | 0.1045 | 0.750 | 0.824 | 0.873 | 1.000 | 36/108 |
| gemm_wide.test2 | 108/11 | 0.6676 | 0.843 | 0.470 | 0.725 | 0.885 | 36/108 |
| flashattention.test1 | 54/6 | 0.1159 | 0.736 | 0.811 | — | 1.000 | 8/54 |
| flashattention.test2 | 54/6 | 0.3798 | 0.711 | 0.711 | — | 0.913 | 8/54 |
| flashattention_causal.test1 | 54/6 | 0.0865 | 0.766 | 0.808 | — | 1.000 | 8/54 |
| flashattention_causal.test2 | 54/6 | 0.2565 | 0.771 | 0.799 | — | 1.000 | 8/54 |
| flashattention_bf16.test1 | 54/6 | 0.1158 | 0.705 | 0.814 | — | 1.000 | 8/54 |
| flashattention_bf16.test2 | 54/6 | 0.3792 | 0.707 | 0.715 | — | 0.920 | 8/54 |
| kda_recurrent.test1 | 6/1 | 0.4883 | — | 1.000 | — | 1.000 | 0/6 |
| kda_recurrent.test2 | 6/1 | 0.9705 | — | 1.000 | — | 1.000 | 0/6 |
| kda_chunk_o.test1 | 24/3 | 0.0150 | 1.000 | 0.979 | — | 0.976 | 8/24 |
| kda_chunk_o.test2 | 24/3 | 0.0220 | 0.990 | 0.981 | — | 1.000 | 8/24 |
| softmax.test1 | 6/1 | 0.0527 | — | 1.000 | — | 0.880 | 0/6 |
| softmax.test2 | 6/1 | 0.1772 | — | 0.920 | — | 0.712 | 0/6 |
| rmsnorm.test1 | 6/1 | 0.0487 | — | 0.997 | — | 0.997 | 0/6 |
| rmsnorm.test2 | 6/1 | 0.1705 | — | 0.996 | — | 0.996 | 0/6 |
| reduce_sum.test1 | 6/1 | 0.0363 | — | 0.990 | — | 0.982 | 0/6 |
| reduce_sum.test2 | 6/1 | 0.0969 | — | 1.000 | — | 0.999 | 0/6 |
| elementwise.test1 | 6/1 | 0.0496 | 1.000 | 0.994 | — | 0.985 | 6/6 |
| elementwise.test2 | 6/1 | 0.1706 | 0.988 | 0.987 | — | 0.988 | 5/6 |

## Prediction and scoring diagnosis

**Pipeline coverage is the first Ampere limitation.** Every GEMM test has only 36/108 finite eligible timing scores: stage 0 is represented, whereas stages 2 and 3 are marked `positive-stage pipeline scheduling policy is unresolved or unsupported` in [pipeline.py](../tilelang/tiletune/pipeline.py). Of 16 GEMM oracle winners, 16 were unscored; 12 used stage 2 and 4 used stage 3. The median GEMM Spearman correlation on the scored subset is 0.863. This correlation says nothing about the unscored two-thirds of the grid. Within the scored subset, the shortlist contains its fastest measured candidate in 16/16 GEMM tests. This separates ranking within modeled schedules from missing schedule coverage.

A seeded offline sanity check draws 500 random shortlists per case at the same K. On GEMM the geometric mean of the per-case random mean Oracle@K is 0.870, compared with timing's 0.818; timing falls below the random mean on 10/16 cases. This diagnostic uses the oracle only after rankings are frozen and never changes selection.

**Absolute latency and ranking are separate tests.** The median across GEMM cases of each case's median predicted/measured latency is 2.58×. The plot retains only scored, successfully measured pairs, so it must be read alongside coverage. Traffic and Carver supply byte-wave scores rather than absolute latency estimates; their scores are evaluated through ranking and Oracle@K.

![Predicted versus measured latency](../experiments/results/ampere-full-comparison-idle/prediction-scatter.png)

The timing model charges byte service using fixed primitive rates. [The primitive profile](../experiments/results/ampere-full-comparison-idle/ampere/primitive-profile.json) records streaming memory service 10.944 bytes/cycle/SM, cached global service 24.801, and shared service 104.585. For clean `gemm_nn.test1`, candidate 0 (32×32×32, stage 0, 128 threads) predicts 0.923 ms versus 0.319 ms measured; copy service accounts for 89.0% of its modeled iteration. Its two unique inputs occupy 16 MiB, versus 1 GiB of logical repeated CTA input loads; the device profile reports 40 MiB L2. Serving those logical input bytes entirely at the measured streaming rate would take 0.644 ms, or 0.284 ms at the cached rate, before other work. These are diagnostic service calculations, not replacement latency predictions. Cache flushing between kernel invocations does not remove reuse among CTAs within an invocation. This arithmetic supports investigating repeated-load/cache accounting; it does not prove a cause without cache/traffic counters and isolated component experiments. Details are in [detail-diagnostics.json](../experiments/results/ampere-full-comparison-idle/detail-diagnostics.json). No workload-specific latency anchor, fitted correction factor or post-test rate adjustment was applied here.

**Reduction and recurrence ownership remain incomplete.** [compute.py](../tilelang/tiletune/compute.py) needs a known fragment producer or explicit layout to infer reduction ownership. Generic softmax, RMSNorm and sum reductions report `no known fragment producer for reduction`; RMSNorm also has unresolved operation work. Recurrent KDA additionally requires a per-iteration schedule for direct scalar global reads. Attention timing scores only 8/54 candidates because positive-stage scheduling and some inter-warp reductions are unresolved. Its median Spearman correlation on scored successful pairs is -0.217, and its six-case Oracle@K geometric mean is 0.732. This ranking result is weaker than GEMM's within-stage result and is based on a small scored subset. Finite latency predictions also do not guarantee that the compiler can realize a layout: XGBoost and traffic selections can spend budget on the known attention compilation failures. For `flashattention.test1`, only 2/6 XGBoost selections and 3/6 traffic selections compiled and measured successfully, versus 6/6 timing selections. A near-oracle winner can therefore coexist with substantial wasted compilation budget. Exact counts remain in each case's diagnostics.

Chunk-KDA and elementwise results cover narrower models and small grids. Near-oracle selection in these cases does not establish calibration or a general model for reductions or complete attention schedules.

## Shape generalization and overfitting

The audit found no source mismatch, no changed frozen ranking, and no model-file mismatch: 38 ranking snapshots (34 supported cases and four empty unsupported cases) were written before their held-out oracle request; 38 XGBoost requests used the frozen model hash. The eight XGBoost models were fitted only from whole-shape training data, with validation early stopping. Test workload identities are rejected by the predictor if they occur in either training or validation. The clean repeat preserves the original model settings even though the preliminary mixed-load run had already been inspected.

XGBoost uses log-latency regression, histogram trees, depth 6, learning rate 0.05, up to 200 rounds, validation patience 20 and seed 123, with per-shape training weights. These are the baseline's pre-existing defaults, not settings selected from these tests. This table reports shape-macro RMSE in natural-log milliseconds and candidate-median absolute errors:

| Operation | Train / val / test shapes | Train log-RMSE | Val log-RMSE | Test log-RMSE | Test prediction / latency | Test median absolute relative error |
| --- | --- | --- | --- | --- | --- | --- |
| attention | 6 / 3 / 6 | 0.012 | 0.459 | 1.640 | 0.221 | 77.9% |
| elementwise | 2 / 1 / 2 | 0.002 | 0.594 | 1.779 | 0.222 | 77.8% |
| gemm | 16 / 8 / 16 | 0.043 | 0.922 | 2.898 | 0.078 | 92.2% |
| kda_chunk_o | 2 / 1 / 2 | 0.002 | 0.113 | 0.499 | 0.608 | 39.2% |
| kda_recurrent | 2 / 1 / 2 | 0.001 | 0.398 | 1.083 | 0.379 | 62.1% |
| reduce_sum | 2 / 1 / 2 | 0.002 | 0.484 | 1.446 | 0.284 | 71.6% |
| rmsnorm | 2 / 1 / 2 | 0.001 | 0.633 | 1.801 | 0.219 | 78.1% |
| softmax | 2 / 1 / 2 | 0.002 | 0.801 | 1.916 | 0.210 | 79.0% |

XGBoost fits training latencies closely, but GEMM log-RMSE rises from 0.043 in training to 2.898 on tests despite its strong GEMM shortlist quality. Its larger softmax test retains only 71.2% of oracle performance. These results show poor absolute scale generalization and uneven selection quality across operations.

A disjoint split prevents label leakage; it does not prove absence of overfitting. This experiment has one device, one data seed, the same kernel implementations/configuration families in all splits, and only two held-out scales. Test shapes extrapolate beyond training sizes. A high test error can reflect scale extrapolation, feature/model limitations or fitting behavior; the available data do not isolate these causes. Good shortlist quality can coexist with poor absolute latency prediction. Additional aspect ratios, irregular sizes, head dimensions, seeds and an untouched final test distribution are needed before a broad generalization claim.

TileTune has no workload-label fitting step in this run. Its fixed primitive measurements and unchanged equations avoid fitting to these test labels, but coverage and measured errors still determine whether its predictions are useful. It should not be described as “overfit-free” or fully validated on Ampere from this matrix alone.

The next model work should prioritize Ampere stage-2/3 asynchronous-copy scheduling, explicit reduction ownership and recurrence accesses, then cache-aware accounting of repeated CTA loads. Attention also needs better ranking and compiler-feasibility handling. Those changes should be checked on an additional untouched workload distribution; they were not implemented or fitted to this test set.

## Cost and reproducibility

| Operation | Training / validation samples | Boosting rounds | Collection seconds (train + validation) | Fit seconds |
| --- | --- | --- | --- | --- |
| attention | 162 / 81 | 199 | 529.9 | 0.117 |
| elementwise | 12 / 6 | 200 | 9.7 | 0.063 |
| gemm | 1728 / 864 | 175 | 1878.8 | 0.222 |
| kda_chunk_o | 48 / 24 | 178 | 65.7 | 0.077 |
| kda_recurrent | 12 / 6 | 200 | 17.5 | 0.063 |
| reduce_sum | 12 / 6 | 200 | 11.8 | 0.067 |
| rmsnorm | 12 / 6 | 200 | 12.1 | 0.055 |
| softmax | 12 / 6 | 200 | 15.9 | 0.066 |

XGBoost training/validation collection totals 2541.2 seconds, plus 0.731 seconds fitting. TileTune primitive preparation took 172.9 seconds. Online tuning speedups exclude those reusable preparation costs, process startup, input/reference construction and final winner remeasurement. They should not be presented as total first-use speedups. Per-case `worker_wall_seconds`, tuning components and model collection provenance are retained in the raw files.

The build used CUDA 12.4, GCC 10 for JIT compilation, PyTorch 2.7.0+cu126 and the A100 `sm_80` target (108 SMs). The native development libraries were rebuilt successfully with CMake. The default GCC 9 caused NVCC to ignore C++20; selecting `/usr/bin/g++-10` resolves it. PyTorch 2.7 omitted some CUDA-property fields used by TileTune, so [device.py](../tilelang/tiletune/src/device.py) now queries missing fields from the matching CUDA runtime device, retaining unknown values instead of fabricating limits. A regression test covers this fallback and the architecture guard.

Verification completed: **373 passed, 13 skipped, 1 deselected**, with 15 warnings. This is the TileTune and experiment test selection, not the entire TileLang test suite. The excluded `blackwell_mma_probe_cross_compiles` needs a CUDA toolkit capable of `sm_100a`, which CUDA 12.4 cannot compile. Ruff, formatting checks, shell syntax and `git diff --check` also passed. Build and test logs are retained in the result directory.

Reproduce the clean full matrix on this machine in a fresh output directory:

```bash
export CUDA_HOME=/root/cuda-12.4
export CXX=/usr/bin/g++-10
export PYTHON=/root/tilelang/.venv/bin/python
export CMAKE_COMMAND=/root/miniconda3/envs/minf/bin/cmake
bash experiments/portable/run_accelerator.sh --build \
  --device ampere --workers 8 --warmup 10 --rep 50 \
  --validation-repeats 7 --wait-idle \
  --output experiments/results/a100-repeat
```

The general [runner](../experiments/portable/run_accelerator.sh) accepts an explicit accelerator manifest and workload/configuration overrides. Native CUDA/HIP workers share the request/result protocol; Carver's adapter currently requires CUDA and supported GEMM semantics. Ascend needs an external compiler worker and its own winner-remeasurement implementation. No other accelerator's runtime performance was tested in this task. See the [portable experiment guide](../experiments/portable/README.md) for manifests, planning, resume checks and support boundaries.

Raw experiment output is retained locally under `experiments/results/` and ignored by Git. The report and runner changes are uncommitted. The earlier local New Carver work remains preserved in `stash@{0}` (`pre-pull-dev-2026-09-13-local-new-carver-work`).

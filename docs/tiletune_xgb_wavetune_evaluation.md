# Sampled XGBoost baseline versus TileTune on A100

Fresh NVIDIA A100 80GB PCIe measurements, FP16, 16 held-out cases across eight workload variants. This is a small shape-extrapolation study on one device.

XGBoost: maximum 600 rounds, depth 10, learning rate 0.05, per-round subsample 0.8, validation patience 20, seed 123. Training shapes use scales 0.25 and 0.5, validation 0.75, and tests 1 and 2. Only a frozen 10% configuration subset per training/validation shape was collected. Failed samples were not replaced. Models were frozen before the test measurements.

TileTune uses pipeline_time and a freshly measured fixed primitive profile. Online budgets are identical: 11/108 configs for GEMM, 6/54 for attention, 3/24 for chunk KDA, and 1/6 for row kernels. Every selected candidate is checked for correctness. Rankings are frozen before each exhaustive reference. Winners are remeasured in seven shuffled rounds.

## Overall

| Metric | XGBoost | TileTune |
| --- | ---: | ---: |
| Oracle@K, geometric mean | 96.59% | 96.86% |
| Oracle@1, diagnostic geometric mean | 93.65% | 84.47% |
| Remeasured winner / oracle performance | 97.09% | 97.12% |
| Online tuning speedup over exhaustive | 3.74 | 2.58 |
| Offline preparation, seconds | 209.92 | 197.26 |
| Online tuning total, seconds | 177.28 | 247.85 |
| Offline + online total, seconds | 387.19 | 445.12 |

Winner performance is effectively tied overall: the repeated-measurement geometric means differ by less than 0.1%. XGBoost uses 28.5% less online tuning time and 13.0% less preparation-plus-online time across this suite. These differences describe the measured run, not a statistical guarantee across devices or workload distributions.

GEMM results depend on size: XGBoost wins the three smaller cases, and TileTune wins the three larger cases. Standard attention is within 1% in repeated timings. TileTune is about 8% faster on causal attention. KDA is close; RMSNorm chooses identical winners. The larger softmax case favors XGBoost by 8.6%.

Oracle@1 is a diagnostic using only the first predicted configuration; it does not change the actual online measurement budgets. All 16 first choices have successful exhaustive measurements. The row models distinguish only two score levels across six configurations, so ties still influence their choices.

The median within-case remeasurement range is 2.6% for XGBoost and 1.1% for TileTune; the maxima are 14.5% and 4.5%. Seven-repeat medians are used, and close timing differences should be treated cautiously.

## By workload variant

Percentages are retained oracle performance (Oracle@K); 100% is best. Each row aggregates the two test scales geometrically.

| Workload | XGBoost | TileTune | XGB speedup over TileTune, remeasured |
| --- | ---: | ---: | ---: |
| gemm_nn | 96.74% | 96.87% | 1.006× |
| gemm_bias_relu | 97.76% | 97.70% | 1.002× |
| gemm_tall | 97.71% | 95.81% | 1.005× |
| flashattention | 93.21% | 92.76% | 1.007× |
| flashattention_causal | 92.06% | 100.00% | 0.925× |
| kda_chunk_o | 96.51% | 96.51% | 1.016× |
| softmax | 99.52% | 96.00% | 1.040× |
| rmsnorm | 99.47% | 99.47% | 1.000× |

## Individual cases

The speedup uses repeated winner measurements: values above 1 favor XGBoost. Tuning times include selection, compilation and shortlist measurement.

| Case | XGB Oracle@K | TileTune Oracle@K | XGB winner µs | TileTune winner µs | XGB speedup | XGB tune s | TileTune tune s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm_nn.test1 | 100.00% | 93.83% | 99.72 | 107.52 | 1.078× | 12.89 | 13.81 |
| gemm_nn.test2 | 93.59% | 100.00% | 737.46 | 692.24 | 0.939× | 13.82 | 14.14 |
| gemm_bias_relu.test1 | 99.43% | 95.45% | 102.35 | 105.69 | 1.033× | 13.78 | 15.88 |
| gemm_bias_relu.test2 | 96.11% | 100.00% | 713.39 | 693.16 | 0.972× | 13.90 | 16.64 |
| gemm_tall.test1 | 100.00% | 91.79% | 103.42 | 109.83 | 1.062× | 12.99 | 13.89 |
| gemm_tall.test2 | 95.47% | 100.00% | 725.12 | 689.25 | 0.951× | 14.21 | 14.22 |
| flashattention.test1 | 89.73% | 89.08% | 129.40 | 130.33 | 1.007× | 20.73 | 31.36 |
| flashattention.test2 | 96.82% | 96.58% | 417.34 | 420.13 | 1.007× | 20.59 | 31.90 |
| flashattention_causal.test1 | 91.54% | 100.00% | 93.06 | 85.70 | 0.921× | 15.10 | 27.94 |
| flashattention_causal.test2 | 92.59% | 100.00% | 283.31 | 263.22 | 0.929× | 14.62 | 27.83 |
| kda_chunk_o.test1 | 97.13% | 97.13% | 15.30 | 15.30 | 1.000× | 6.85 | 7.63 |
| kda_chunk_o.test2 | 95.89% | 95.89% | 22.15 | 22.88 | 1.033× | 6.47 | 8.00 |
| softmax.test1 | 99.05% | 100.00% | 53.13 | 52.95 | 0.997× | 2.91 | 6.85 |
| softmax.test2 | 100.00% | 92.15% | 177.13 | 192.31 | 1.086× | 3.17 | 8.15 |
| rmsnorm.test1 | 99.28% | 99.28% | 48.94 | 48.94 | 1.000× | 2.62 | 4.58 |
| rmsnorm.test2 | 99.66% | 99.66% | 170.79 | 170.79 | 1.000× | 2.63 | 5.04 |

## Training data and preparation

Selected counts include failed configurations; labels count only successful measurements. Preparation includes fresh training/validation collection and CPU fitting.

| Model | Train selected / pool | Train labels | Validation selected / pool | Validation labels | Best rounds | Preparation s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention | 24/216 | 11 | 12/108 | 5 | 152 | 75.65 |
| gemm | 66/648 | 66 | 33/324 | 33 | 332 | 100.76 |
| kda_chunk_o | 6/48 | 6 | 3/24 | 3 | 212 | 17.84 |
| rmsnorm | 2/12 | 2 | 1/6 | 1 | 225 | 7.24 |
| softmax | 2/12 | 2 | 1/6 | 1 | 1 | 8.42 |

## Interpretation limits

- Some row models have only two training labels and one validation label under the requested 10% sampling policy; their results do not establish robust generalization.
- The configuration spaces are small for row kernels. Full-grid measurement can be cheaper than analysis plus selection there.
- Oracle@K assesses the shortlist using one common exhaustive measurement table. Remeasured performance assesses the actual selected winner and can differ because of measurement noise or a different within-shortlist choice.
- Per-method costs exclude process startup, reference construction, exhaustive evaluation references, and winner remeasurement. Offline model/profile preparation is shown separately and in the combined totals.
- No settings were adjusted using these test results. The older full-pool XGBoost performance reports use a different training protocol.
- Concurrent shared-workspace edits triggered the XGBoost source guard. The affected larger fused-GEMM and first tall-GEMM cases were invalidated and rerun from the original source snapshot in an isolated checkout. The final included experiments pass source/native-build consistency checks; discarded records and the isolation details remain available in this directory.

![Per-case capability comparison](../experiments/results/ampere-xgb-wavetune-20260914/capability.png)

Raw records: [comparison.json](../experiments/results/ampere-xgb-wavetune-20260914/comparison.json), [results-summary.json](../experiments/results/ampere-xgb-wavetune-20260914/results-summary.json), [plan.json](../experiments/results/ampere-xgb-wavetune-20260914/plan.json), [provenance.json](../experiments/results/ampere-xgb-wavetune-20260914/provenance.json), [isolation.json](../experiments/results/ampere-xgb-wavetune-20260914/isolation.json).

Two worker timings that overlapped external GPU contexts were preserved in `invalidated-by-contention` and rerun. Their rankings exactly match the originals; continuous monitoring detected no overlap after the clean restart. See [contention-reruns.json](../experiments/results/ampere-xgb-wavetune-20260914/contention-reruns.json). Elapsed experiment time additionally includes GPU waiting, discarded attempts, reference generation, exhaustive evaluation, and winner remeasurement.

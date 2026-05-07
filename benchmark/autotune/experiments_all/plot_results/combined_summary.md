# Experiments Summary

- Generated from 9 CSV files.
- Each section uses its own baseline row for speedup.

## Grouped Compilation

| label | end_to_end_s | compilation_s | benchmark_s | best_latency_ms | e2e_speedup |
|---|---:|---:|---:|---:|---:|
| G1 | 80.373 | 69.720 | 10.649 | 0.5326 | 1.000 |
| G2 | 50.726 | 37.768 | 12.954 | 0.5320 | 1.584 |
| G4 | 43.553 | 26.259 | 17.290 | 0.5323 | 1.845 |

## Pipeline Only

| label | end_to_end_s | compilation_s | benchmark_s | best_latency_ms | e2e_speedup |
|---|---:|---:|---:|---:|---:|
| NoPipe | 79.258 | 68.577 | 10.677 | 0.5324 | 1.000 |
| Pipe | 69.742 | 69.170 | 62.765 | 0.5320 | 1.136 |

## Multi-GPU Benchmark

| label | end_to_end_s | compilation_s | benchmark_s | best_latency_ms | e2e_speedup |
|---|---:|---:|---:|---:|---:|
| 1GPU | 79.358 | 68.700 | 10.655 | 0.5331 | 1.000 |
| MultiGPU | 73.057 | 67.685 | 5.368 | 0.5332 | 1.086 |

## All Features Combined

| label | end_to_end_s | compilation_s | benchmark_s | best_latency_ms | e2e_speedup |
|---|---:|---:|---:|---:|---:|
| Baseline | 222.513 | 76.233 | 146.203 | 0.1757 | 1.000 |
| AllFeat | 138.225 | 52.653 | 120.412 | 0.1762 | 1.610 |


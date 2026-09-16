# H200 expanded-pool results

Completed on 2026-09-16: all 2,304 configurations for each final FP16 GEMM
shape, using all idle H200s with contention monitoring. Each shape has 1,920
correct benchmarked candidates and 384 recorded WGMMA compilation failures
(`block_M=96`). Every original example configuration is included.

| Record | Winning tile M×N×K | Paired FP16 TFLOP/s | Previous example TFLOP/s | Result |
| --- | --- | --- | --- | --- |
| [4096³](gemm_square.json) | 128×256×64 | 648.78 | 650.36 | Same config/code; matches within noise |
| [8192³](gemm_square_large.json) | 192×256×64 | 664.22 | 643.13 | +3.36% median paired speedup |

Both winners use 3 stages, 256 threads and `enable_rasteration=True`.
The comparison uses 248 paired rounds across six uncontended GPUs, with
25 ms warmup and a 200 ms repetition budget. Original sweep minima were
731.68 and 706.62 TFLOP/s; raw minima, the initial seven validation samples,
and the longer comparisons are retained separately in the JSON records.

Both sweeps plus initial validation took approximately 22.1 active minutes
(26.8 wall minutes including a coordinator interruption), on 6–8 available
GPUs. Two contended sweep shards were discarded and retried. Monitoring
records absence of observed foreign compute processes; it cannot exclude
interference between observations.

See the [full report](../../../results/h200-gemm-expanded-2304-20260916-v1/report.md)
for exact timing settings, comparison with both example baselines, all failures,
per-GPU variation, provenance and raw logs. Previous records are archived under
`experiments/results/gemm-pre-single-pool-20260916/heuristics/`.

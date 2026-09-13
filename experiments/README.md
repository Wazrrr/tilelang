# Final autotuning experiments

For the expanded workload × accelerator matrix, see
[portable experiments](portable/README.md). It covers GEMM variants,
FlashAttention, KDA, normalization/reduction kernels, native CUDA/HIP execution,
and the external worker boundary for Huawei Ascend. The fixed-grid comparisons
below remain separate entry points.

The optional [XGBoost baseline](xgboost/README.md) trains on separate exhaustive
workloads and selects a frozen top-K on held-out shapes. It is available in the
portable matrix and the GEMM, FP8, and FlashAttention comparison runners through
`--method xgboost --xgb-model MODEL`; supplying a model also adds it to their
`--method all` comparison.

Run these commands from the repository root in an environment with TileLang,
PyTorch, and CUDA. Each entry point runs an experiment and writes its results.

```text
experiments/
├── gemm/
│   ├── kernel.py             Advanced-autotune GEMM and its 288-config grid
│   ├── system/run.py         Pipeline, grouped compilation, multi-GPU comparison
│   └── tiletune/run.py       Brute force / Carver / TileTune / XGBoost top-k
├── gemm_fp8/
│   ├── kernel.py             FP8 example adapter for concurrent compilation
│   ├── system/run.py         Five system variants, E4M3 or E5M2
│   └── tiletune/run.py       Brute force / TileTune top-k, 288 configs
├── flash_attention/
│   ├── kernel.py             Attention construction, inputs, and correctness
│   ├── system/run.py         Five system variants, causal or noncausal
│   └── tiletune/run.py       Brute force / TileTune top-k, 128 configs
├── portable/                Workload × accelerator matrix and worker protocol
├── xgboost/                 Offline training, frozen selection, and evaluation
├── _common.py               Run arguments, result files, compile outcomes
└── _tiletune.py             TileTune arguments and model-rank reporting
```

GEMM uses the tiled `A @ B.T` kernel from the advanced autotune example, with
FP32 accumulation and a shared-memory epilogue. Both GEMM experiment types use
the same kernel and grid. FP8 and FlashAttention elaborate the existing example
kernels through their JIT interfaces.

## Experiment commands by kernel

Each kernel folder documents its own runnable cases:

| Kernel | Experiment commands |
| --- | --- |
| [GEMM](gemm/README.md) | Five system cases: baseline, pipeline, grouped compilation, multi-GPU, and combined; FP16/BF16 TileTune runs |
| [FP8 GEMM](gemm_fp8/README.md) | Five system cases and brute-force/TileTune comparisons for E4M3 and E5M2 |
| [FlashAttention](flash_attention/README.md) | Five system cases and brute-force/TileTune comparisons for causal and noncausal attention |

## TileTune ranking experiments

Each kernel has its own system and TileTune runner. System `--variant all` runs
baseline, pipeline, grouped compilation, multi-GPU, and combined variants in
fresh processes. These cases disable TileTune and tune the entire supplied grid.
They default to the `event` backend and require two visible GPUs for `all`.

The GEMM runner compares brute force, Carver, and TileTune; FP8 and FlashAttention
compare brute force and TileTune. `--method all` runs the comparison in separate
processes, with model selection preceding exhaustive measurements. Each TileTune
runner defaults to a fixed `--top-k 20`, a reusable device profile, and
**`ranking_metric="pipeline_time"`**. At most K scored, eligible configurations
are compiled and benchmarked, with no replacement after failures. Brute force
tunes the entire supplied grid. Correctness checks remain enabled and all
candidate outcomes are retained.

FP8 and FlashAttention also accept `--method tiletune --top-k all` for their
previous exhaustive ranking experiment. This uses `report_only` mode and
`early_stop=False`: every successfully analyzed candidate proceeds to
compilation, including candidates the model would reject for pressure.

Comparisons remeasure the selected winners in shuffled order, report the median
of 5 samples, and save every sample. They also report Oracle@K: exhaustive best
latency divided by the best exhaustive latency within the frozen selected set.
Coverage excludes candidates without successful exhaustive measurements. Profile
preparation and final validation are excluded from reported tuning time;
top-K analysis and selection are included.

Primitive rates are measured or loaded before candidate timing. Candidate
latencies are used to identify the measured winner, not to fit the model or
change its ranking. Device profiling time is reported separately from tuning.
The existing profiler supports A100 and Hopper; FP8 requires Hopper.

## Read the winner's rank

Each TileTune winner summary records latency, original configuration index,
predicted rank, and the complete tie interval. For example (illustrative values):

```json
{
  "latency_ms": 0.123456,
  "original_index": 42,
  "predicted_rank": 7,
  "tie_first_rank": 7,
  "tie_last_rank": 8
}
```

Ranks are one-based and refer to the model's ordering of the **supplied grid**.
Ties show the full possible rank interval. If the measured winner is unscored or
marked pressure-rejected by the model, the summary uses null predicted ranks
with its tier and report position. An appended position for an unknown result
is not presented as a predicted performance rank.

`--memory-regime streaming` is the default model input; `cached` selects the
profile's cached-memory rates. TileTune runners default to CUDA graphs. Omitting
`--device-profile` uses the profiler's automatic cache. An explicit path is
created if needed and can accumulate FP16, BF16, and FP8 primitive measurements
for the same device/build fingerprint. Use a separate path after an incompatible
device or build change.

## Run sizes and output files

All runners accept `--workers`, `--warmup`, `--rep`, `--timeout`, and `--seed`.
TileTune accepts `--group-size` (default 1, grouping disabled), `--method`, and
`--top-k` (default 20); its library top-k option is
`TileTuneConfig(top_k=...)`, with `None` preserving exhaustive behavior.
`--config-indices 0 8 16 24` runs an explicit subset for a shorter experiment;
the default is the entire grid. Subset reports preserve original indices and
label their grid size. They do not claim an exhaustive full-grid winner.

`--output` is a reusable parent directory. Each invocation creates a new UTC
timestamp subdirectory, prints its path at startup, and writes the files below
there. Repeat the same command to keep multiple versions without overwriting
existing results. Existing files directly under `--output` are left in place.

For a named version, add `--run-name v2`; results go to `<output>/v2/`.
An existing run name is rejected, even if its directory is empty.
For system `--variant all`, one run directory contains `comparison.json` and
the five variant subdirectories. Generated results under `experiments/results/`
and profiles under `experiments/profiles/` are ignored by Git.

| File | Contents |
| --- | --- |
| `experiment.json` | Arguments, devices, original indices, grid, and TileTune's fixed profile/source hashes |
| `benchmarks.tsv` | Per-candidate benchmark outcomes and latencies |
| `timings.tsv` | Compilation and autotuning stage measurements |
| `summary.json` | Tuning duration and measured winner; TileTune also records its predicted rank and ties |
| `outcomes.json` | Comparison method's candidate outcomes, original indices, selection flags, and failures |
| `tiletune.json` | TileTune's analysis, resource decisions, ranking, and outcome for every supplied candidate |
| `comparison.json` | System `--variant all` or TileTune `--method all` results, speedups, and comparison metrics |

# Final autotuning experiments

Run these commands from the repository root in an environment with TileLang,
PyTorch, and CUDA. Each entry point runs an experiment and writes its results.

```text
experiments/
├── gemm/
│   ├── kernel.py             Advanced-autotune GEMM and its 288-config grid
│   ├── system/run.py         Pipeline, grouped compilation, multi-GPU comparison
│   └── tiletune/run.py       Brute force / Carver / TileTune top-k comparison
├── gemm_fp8/
│   ├── kernel.py             FP8 example adapter for concurrent compilation
│   └── tiletune/run.py       FP8 GEMM, 288 configs
├── flash_attention/
│   └── tiletune/run.py       FlashAttention, 128 configs, causal or noncausal
├── _common.py               Run arguments and result files
└── _tiletune.py             Shared TileTune experiment and rank reporting
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
| [FP8 GEMM](gemm_fp8/README.md) | E4M3 and E5M2 TileTune runs |
| [FlashAttention](flash_attention/README.md) | Noncausal and causal TileTune runs |

## TileTune ranking experiments

The GEMM runner compares brute force, Carver, and TileTune with a fixed top-k;
see its [kernel guide](gemm/README.md). FP8 and FlashAttention runs obtain a
reusable device profile, use **`ranking_metric="pipeline_time"`**,
and exhaustively tune the supplied grid with `early_stop=False`. The mode is
`report_only`: pressure decisions are recorded, and every successfully analyzed
candidate proceeds to compilation. Correctness checks remain enabled. Failed
candidates stay in `tiletune.json`; the winner is the fastest successful candidate.

Primitive rates are measured or loaded before candidate timing. Candidate
latencies are used to identify the measured winner, not to fit the model or
change its ranking. Device profiling time is reported separately from tuning.
The existing profiler supports A100 and Hopper; FP8 requires Hopper.

## Read the winner's rank

FP8 and FlashAttention runs end with lines in this form (illustrative values);
GEMM prints the same rank information in its winner summary and comparison:

```text
Measured winner: 0.123456 ms, original config #42
Winner config: {...}
Winner's pipeline_time rank: 7/288 (tie range 7-8; 240 scored candidates)
```

Ranks are one-based and refer to the model's ordering of the **supplied grid**.
Ties show the full possible rank interval. If the measured winner is unscored or
marked pressure-rejected by the model, the script prints `rank: unavailable`
with its tier and report position. An appended position for an unknown result
is not presented as a predicted performance rank.

`--memory-regime streaming` is the default model input; `cached` selects the
profile's cached-memory rates. Both benchmark with CUDA graphs. Omitting
`--device-profile` uses the profiler's automatic cache. An explicit path is
created if needed and can accumulate FP16, BF16, and FP8 primitive measurements
for the same device/build fingerprint. Use a separate path after an incompatible
device or build change.

## Run sizes and output files

All runners accept `--workers`, `--warmup`, `--rep`, `--timeout`, and `--seed`.
TileTune accepts `--group-size` (default 1, grouping disabled). GEMM additionally
accepts `--method` and `--top-k` (default 20); its library top-k option is
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
| `tiletune.json` | TileTune's analysis, resource decisions, ranking, and outcome for every supplied candidate |
| `comparison.json` | System `--variant all` results and speedups; each variant has its own subdirectory |

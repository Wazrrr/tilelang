# GEMM experiments

Run all commands from the repository root in an environment with TileLang,
PyTorch, and CUDA. Each command creates a timestamped run subdirectory under
`--output`, so the same command can be repeated to retain multiple versions.
Add `--run-name v2` for a named version; an existing name is rejected.

Both experiment types use [kernel.py](kernel.py): the advanced-autotune
`A @ B.T` GEMM, with FP32 accumulation, a shared-memory epilogue, and the same
288 configurations. The system experiments default to BF16; New Carver supports
both FP16 and BF16.

## System experiments: five cases

[system/run.py](system/run.py) measures autotuning wall time and the measured
winner's kernel latency. New Carver and legacy filtering are disabled.

| Case | Compile/benchmark pipeline | Grouped compilation | Multi-GPU benchmarking |
| --- | --- | --- | --- |
| `baseline` | Off | Off | Off |
| `pipeline` | On | Off | Off |
| `grouped` | Off | On | Off |
| `multi_gpu` | Off | Off | On |
| `combined` | On | On | On |

The following Bash commands run each case separately on the full grid:

```bash
# 1. Baseline: independent compilation, then single-GPU benchmarking.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm.system.run \
    --variant baseline --benchmark-devices 0 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --output experiments/results/gemm/system/baseline

# 2. Pipeline: overlap compilation with single-GPU benchmarking.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm.system.run \
    --variant pipeline --benchmark-devices 0 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --output experiments/results/gemm/system/pipeline

# 3. Grouped compilation: compile two configurations per unit.
CUDA_VISIBLE_DEVICES=2 python -m experiments.gemm.system.run \
    --variant grouped --benchmark-devices 0 --group-size 2 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --output experiments/results/gemm/system/grouped

# 4. Multi-GPU: distribute candidate benchmarks across two GPUs.
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemm.system.run \
    --variant multi_gpu --benchmark-devices 0 1 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --output experiments/results/gemm/system/multi_gpu

# 5. Combined: pipeline, grouped compilation, and multi-GPU benchmarking.
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemm.system.run \
    --variant combined --benchmark-devices 0 1 --group-size 2 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --output experiments/results/gemm/system/combined
```

Alternatively, run the five cases in fresh processes and automatically print a
comparison table. This command uses a separate output directory from the cases
above:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemm.system.run \
    --variant all --benchmark-devices 0 1 --group-size 2 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --output experiments/results/gemm/system_comparison
```

`--benchmark-devices` names logical CUDA ordinals after visibility is applied.
For example, with `CUDA_VISIBLE_DEVICES=4,6`, use `--benchmark-devices 0 1`.
Single-GPU cases use the first listed device. Multi-GPU cases require at least
two devices; use GPUs of the same model for timing comparisons.

The autotuner pipeline overlaps compilation with benchmarking. The kernel's
`num_stages` is a separate parameter swept by every case. Multi-GPU benchmarking
distributes configurations; it does not split one GEMM across devices.

All five cases use the same grid, configuration order, and seeded inputs.
Defaults are 4 CPU compilation workers, 10 warmup repetitions, 100 benchmark
repetitions, seed 123, and the `event` backend. Kernel and autotune caches are
disabled. Keep these settings equal when comparing cases. The comparison reports
`baseline tuning_seconds / variant tuning_seconds`; process startup and input
creation are outside the tuning interval.

Each case writes `experiment.json`, `benchmarks.tsv`, `timings.tsv`, and
`summary.json` inside its run directory, for example:

```text
experiments/results/gemm/system/grouped/
├── 20260908T120000.123456Z/
│   ├── experiment.json
│   ├── benchmarks.tsv
│   ├── timings.tsv
│   └── summary.json
└── 20260908T130000.654321Z/
    └── ...
```

Existing results directly in `grouped/` stay in place. `--variant all` writes
`comparison.json` inside its run directory alongside `baseline/`, `pipeline/`,
`grouped/`, `multi_gpu/`, and `combined/`.

## New Carver experiments

[new_carver/run.py](new_carver/run.py) uses `pipeline_time` to rank all 288
configurations and prints the measured winner's predicted rank and tie range.
It runs in `report_only` mode with correctness checks and no early stopping.
The winner is selected from the candidates that successfully compile and
benchmark; all candidate outcomes remain in the report.

```bash
# FP16 GEMM.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm.new_carver.run \
    --m 4096 --n 4096 --k 4096 --dtype float16 \
    --device-profile experiments/profiles/h200.json \
    --output experiments/results/gemm/new_carver_fp16

# BF16 GEMM: same datatype as the system experiments.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm.new_carver.run \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --device-profile experiments/profiles/h200.json \
    --output experiments/results/gemm/new_carver_bf16
```

The device profile is measured if needed, then reused. Profiling time is
reported separately from tuning. The model uses streaming-memory rates by
default and benchmarks candidates with CUDA graphs. It supports A100 and Hopper;
use a separate profile path for each incompatible device/build fingerprint.

The final output prints the winner's configuration, latency, and one-based
`pipeline_time` rank. Ties include their full rank interval. An unscored or
pressure-rejected winner is labeled `rank: unavailable`, with its report position.
Results include `carver.json` and `summary.json` with these fields.

## Run options

Both runners accept `--workers`, `--warmup`, `--rep`, `--timeout`, `--seed`, and
`--config-indices`. For example, `--config-indices 0 8 16 24` runs a four-config
subset; its winner and rank refer only to that subset. New Carver's `--group-size`
defaults to 1, disabling grouping; use 2 or more to enable it.

See the [shared experiment guide](../README.md) for profile options, rank
interpretation, and the output-file schema.

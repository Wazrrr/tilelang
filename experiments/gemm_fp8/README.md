# FP8 GEMM experiments

Run all commands from the repository root in an environment with TileLang,
PyTorch, and a Hopper CUDA GPU. Each command creates a timestamped run
subdirectory under `--output`; repeat the command to retain multiple versions.
Add `--run-name v2` for a named version; an existing name is rejected.

[kernel.py](kernel.py) elaborates the existing FP8 `A @ B.T` example, with FP32
accumulation and output in the input datatype. The grid contains 288
configurations. Kernel construction is serialized because the eager builder
mutates shared state; lowering and compilation remain parallel.

## New Carver experiments

[new_carver/run.py](new_carver/run.py) ranks the full grid with `pipeline_time`,
benchmarks every successfully analyzed and compiled candidate, and prints the
measured winner's predicted rank. It uses `report_only` mode, no early stopping,
and the example's `calc_diff < 1e-3` correctness criterion.

```bash
# FP8 E4M3.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm_fp8.new_carver.run \
    --m 4096 --n 4096 --k 4096 --dtype float8_e4m3fn \
    --device-profile experiments/profiles/h200.json \
    --output experiments/results/gemm_fp8/e4m3

# FP8 E5M2.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm_fp8.new_carver.run \
    --m 4096 --n 4096 --k 4096 --dtype float8_e5m2 \
    --device-profile experiments/profiles/h200.json \
    --output experiments/results/gemm_fp8/e5m2
```

The profile reuses common primitive measurements and adds measurements for each
FP8 datatype when needed. Rates are fixed before candidate timing. Candidate
latencies identify the measured winner and do not fit or change the model.
Profiling time is reported separately from tuning. Use a separate profile path
after an incompatible device or build change.

Both cases use streaming-memory model rates and CUDA-graph benchmarking by
default. Defaults are 4 compilation workers, 10 warmup repetitions, 100 benchmark
repetitions, and seed 123. Kernel and autotune caches are disabled.

## Results and options

Each run prints the measured winner's configuration, latency, and one-based
`pipeline_time` rank, including the full tie interval. If the model leaves the
winner unscored or marks it pressure-rejected, the output says
`rank: unavailable` and supplies its report position and tier.

`summary.json` contains the winner and rank. `carver.json` retains the ranking,
analysis, and every candidate outcome, including failures. `experiment.json`
records the grid, input settings, device, fixed profile, and source hashes.
`benchmarks.tsv` and `timings.tsv` record measurements and stage costs.

Use `--group-size 2` for grouped compilation; the default 1 disables grouping.
`--workers`, `--warmup`, `--rep`, `--timeout`, and `--seed` control the run.
`--config-indices 0 8 16 24` selects a four-config subset; the reported winner's
rank then refers to that subset. `--memory-regime cached` selects the profile's
cached-memory rates instead of its streaming rates.

See the [shared experiment guide](../README.md) for profile options and the
output-file schema.

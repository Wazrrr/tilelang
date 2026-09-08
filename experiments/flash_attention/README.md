# FlashAttention experiments

Run all commands from the repository root in an environment with TileLang,
PyTorch, and an A100 or Hopper CUDA GPU. Each command creates a timestamped run
subdirectory under `--output`; repeat the command to retain multiple versions.
Add `--run-name v2` for a named version; an existing name is rejected. Use a
device-profile path appropriate to the selected GPU.

[tiletune/run.py](tiletune/run.py) elaborates the existing FlashAttention
forward example with FP16 BSHD inputs and a chunked FP32 reference. Both causal
and noncausal experiments use the full 128-config grid.

## TileTune experiments

Both cases rank with `pipeline_time`, use `report_only` mode, and disable early
stopping. Every successfully analyzed and compiled candidate is benchmarked with
correctness checks. The measured winner's configuration, latency, predicted
rank, and tie range are printed at the end.

```bash
# Noncausal FlashAttention.
CUDA_VISIBLE_DEVICES=0 python -m experiments.flash_attention.tiletune.run \
    --batch 1 --heads 16 --sequence 4096 --dim 128 \
    --device-profile experiments/profiles/h200-tiletune.json \
    --output experiments/results/flash_attention/noncausal

# Causal FlashAttention.
CUDA_VISIBLE_DEVICES=0 python -m experiments.flash_attention.tiletune.run \
    --batch 1 --heads 16 --sequence 4096 --dim 128 --causal \
    --device-profile experiments/profiles/h200-tiletune.json \
    --output experiments/results/flash_attention/causal
```

The device profile is measured if needed and reused by both cases. Its primitive
rates are fixed before candidate timing; candidate latencies are not used to
fit or change the ranking model. Profiling time is reported separately from
tuning. Use a separate profile path after an incompatible device or build change.

The soft register allowance defaults to 32 registers per computing thread,
matching the existing attention example. Use
`--spill-budget-registers-per-thread 0` for a strict demand comparison. The
allowance never enlarges physical register capacity. Compiler spill and local
memory usage are recorded without imposing byte limits.

## Results and options

Ranks are one-based and refer to the supplied configuration grid. Ties include
the full rank interval. An unscored or pressure-rejected winner is labeled
`rank: unavailable`, with its tier and report position. Failed candidates remain
in the report; the measured winner is the fastest successful candidate.

`summary.json` contains the winner and rank. `tiletune.json` retains the complete
analysis, ranking, and candidate outcomes. `experiment.json` records the workload,
grid, device, fixed profile, and source hashes. `benchmarks.tsv` and `timings.tsv`
record measurements and stage costs.

Defaults are 4 compilation workers, 10 warmup repetitions, 100 benchmark
repetitions, seed 123, streaming-memory model rates, and CUDA-graph benchmarking.
Kernel and autotune caches are disabled. `--workers`, `--warmup`, `--rep`,
`--timeout`, and `--seed` control the run. `--group-size 2` enables grouped
compilation; the default 1 disables it. `--memory-regime cached` selects the
profile's cached-memory rates.

`--config-indices 40 44 46` selects a smaller grid for a shorter run. Its winner
and rank refer only to the selected configurations.

See the [shared experiment guide](../README.md) for profile options and the
output-file schema.

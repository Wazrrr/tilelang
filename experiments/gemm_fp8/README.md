# FP8 GEMM experiments

Run commands from the repository root with TileLang, PyTorch, and Hopper CUDA GPUs.
Every invocation creates a UTC timestamp directory under `--output`. Repeat a
command to retain another run, or use `--run-name v2` for a named version.
Existing run directories are rejected.

Both experiment scripts use [kernel.py](kernel.py). The kernel computes `A @ B.T` with FP32 accumulation and output in the input
datatype. The grid contains 288 configurations. The wrapper converts eager-JIT
return metadata to an explicit output index so independent and grouped
compilation use the same interface.
Kernel construction is serialized; lowering and compilation remain parallel.

## System experiments

[system/run.py](system/run.py) runs the same five cases as the GEMM experiment.
TileTune is disabled in these cases, and the whole supplied grid is tuned.

| Variant | Compile/benchmark overlap | Grouped compilation | Multi-GPU benchmarking |
| --- | --- | --- | --- |
| `baseline` | Off | Off | Off |
| `pipeline` | On | Off | Off |
| `grouped` | Off | On | Off |
| `multi_gpu` | Off | Off | On |
| `combined` | On | On | On |

Each command below runs all five variants sequentially in separate processes:

```bash
# E4M3.
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemm_fp8.system.run \
    --variant all --benchmark-devices 0 1 --group-size 2 \
    --m 4096 --n 4096 --k 4096 --dtype float8_e4m3fn \
    --output experiments/results/gemm_fp8/system_e4m3

# E5M2.
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemm_fp8.system.run \
    --variant all --benchmark-devices 0 1 --group-size 2 \
    --m 4096 --n 4096 --k 4096 --dtype float8_e5m2 \
    --output experiments/results/gemm_fp8/system_e5m2
```

For one case, replace `--variant all` with a variant from the table. Baseline,
pipeline, and grouped cases need only one visible GPU and `--benchmark-devices 0`.
The multi-GPU and combined cases require at least two visible GPUs; ordinals are
logical indices after `CUDA_VISIBLE_DEVICES` is applied. Use matching GPU models.
Multi-GPU benchmarking distributes candidates, not the work of one kernel.

All variants use the same seeded inputs, reference, and correctness checker.
Reference outputs are prepared on each benchmark device before tuning. Caches
are disabled; input tensors are reused by the input supplier. Defaults are
4 compilation workers, 10 warmup repetitions, 100 benchmark repetitions, seed
123, and the `event` backend. Tuning wall time excludes input/reference setup.
The comparison reports baseline wall time divided by each variant's wall time.
Compilation stage work totals are not parallel wall times.

## TileTune comparison

[tiletune/run.py](tiletune/run.py) supports `--method brute_force`, `tiletune`,
and `all`. TileTune defaults to `--top-k 20`. Both methods use the same kernel,
inputs, correctness check, and supplied configuration grid. Carver is not part
of this experiment: its current comparison adapter is specific to FP16/BF16 GEMM.

```bash
# E4M3: compare exhaustive search with TileTune's fixed top-20.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm_fp8.tiletune.run \
    --method all --top-k 20 \
    --m 4096 --n 4096 --k 4096 --dtype float8_e4m3fn \
    --device-profile experiments/profiles/h200-gemm_fp8.json \
    --output experiments/results/gemm_fp8/comparison_e4m3

# E5M2: compare exhaustive search with TileTune's fixed top-20.
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm_fp8.tiletune.run \
    --method all --top-k 20 \
    --m 4096 --n 4096 --k 4096 --dtype float8_e5m2 \
    --device-profile experiments/profiles/h200-gemm_fp8.json \
    --output experiments/results/gemm_fp8/comparison_e5m2
```

`--method all` runs TileTune before brute force, in separate processes. TileTune
measures or loads a compatible device profile, freezes its `pipeline_time`
ranking, then compiles and benchmarks at most K eligible configurations.
Unscored and pressure-rejected candidates are excluded from top-K. Failed
selected candidates remain in the report and are not replaced. Brute force
benchmarks every candidate that compiles successfully.

The chosen winners are recompiled and remeasured in shuffled order with the same
inputs. The comparison reports the median of 5 measurements and saves every
sample; `--validation-repeats` changes this count. Validation never changes the
selected winner. Device-profile preparation and final validation are reported
separately from tuning time; top-K analysis and selection are included in tuning.

The comparison also reports `top_k_oracle_retained_performance` (Oracle@K): the
best exhaustive latency divided by the best exhaustive latency inside the frozen
selected set. This evaluates shortlist quality using one measurement table.
Candidates without successful exhaustive measurements are excluded, and their
coverage is reported. Final validation and Oracle@K use different measurements.

To reproduce the previous exhaustive TileTune ranking experiment, use
`--method tiletune --top-k all`. This keeps `report_only` analysis and benchmarks
all successfully analyzed and compiled configurations, including candidates the
model would reject for pressure. The winner summary retains its predicted rank
and tie interval, or null ranks and its tier when the model cannot score it.

TileTune defaults to CUDA graphs, streaming-memory model rates, and grouped
compilation disabled (`--group-size 1`). Set `--group-size 2` to enable grouping
equally for both methods. `--backend event` and `--memory-regime cached` are
available. Primitive profiles are fixed before candidate timing and reused when
compatible; candidate latencies never fit or alter the model. Use a new profile
path after an incompatible device or native build change.

Correctness uses the FP8 example's `calc_diff < 1e-3` criterion against FP32
matmul rounded to the output dtype. Reported TFLOPS use `2*M*N*K` FLOPs.

## Results and small runs

Every individual run writes `experiment.json`, `summary.json`, `benchmarks.tsv`,
and `timings.tsv`. TileTune comparison methods also write `outcomes.json`, and
the TileTune method writes its complete analysis and ranking to `tiletune.json`.
An `all` run contains method/variant subdirectories and `comparison.json`.
TileTune comparisons additionally save child logs and `validation_timings.tsv`.

`--workers`, `--warmup`, `--rep`, `--timeout`, and `--seed` control run size.
`--config-indices` selects a subset and preserves its original grid indices.
Ranks and the exhaustive winner then refer only to that subset.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.gemm_fp8.tiletune.run \
    --method all --top-k 1 --m 256 --n 256 --k 256 --dtype float8_e4m3fn --config-indices 0 8 \
    --workers 2 --warmup 1 --rep 2 --validation-repeats 2 \
    --output experiments/results/gemm_fp8/smoke
```

See the [shared experiment guide](../README.md) for the directory layout.

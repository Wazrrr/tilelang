# GEMM Autotune End-to-End Benchmark

This benchmark evaluates GEMM autotuning end-to-end in TileLang and records per-case CSV rows.

- Driver: `benchmark/autotune/run_gemm_autotune_e2e.py`
- Kernel source: `examples/gemm/example_gemm_autotune.py`
- Main timing signals:
  - `end_to_end_s`
  - `compilation_s`
  - `benchmark_s`

## Current Default Setup

When no arguments are passed, the script runs:

- shape list: only `square_large = 4096x4096x4096`
- `cpu_count`: `16`
- `runs`: `1`
- `warmup`: `3`
- `rep`: `20`
- `timeout`: `180`
- `with_roller`: `False`
- `skip_check`: `True`
- `disable_cache`: `True`
- output CSV: `results/gemm_autotune_e2e.csv`

## Basic Run

From `benchmark/autotune`:

```bash
python run_gemm_autotune_e2e.py
```

From repo root:

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py
```

## Useful Sweeps

CPU parallelism sweep:

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --cpu-count 1 --cpu-count 4 --cpu-count 8 --cpu-count 16 \
  --runs 3 \
  --csv benchmark/autotune/results/gemm_autotune_e2e_cpu_sweep.csv
```

Custom shape mix:

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --shape 4096x4096x4096 \
  --shape 4096x4096x512 \
  --shape 1024x16384x4096 \
  --csv benchmark/autotune/results/gemm_autotune_custom.csv
```

Enable roller-based search space:

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --with-roller \
  --topk 20
```

## CSV Columns

- `shape_tag`, `shape_class`, `shape`
- `problem_gflop`
- `cpu_count`, `run_idx`
- `status`, `error`
- `num_configs`
- `end_to_end_s`
- `compilation_s`: compile stage time from futures submission to compile collection
- `benchmark_s`: benchmark stage time spent in the benchmark loop
- `best_latency_ms`, `best_tflops`
- `ref_latency_ms`, `ref_tflops`
- `cpu_count_env`
- `execution_backend`, `profile_backend`
- `warmup`, `rep`, `timeout`
- `with_roller`, `topk`, `skip_check`, `cache_input_tensors`
- `best_config` (JSON string)

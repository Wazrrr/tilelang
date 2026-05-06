# GEMM Autotune End-to-End Benchmark

This benchmark evaluates GEMM autotuning end-to-end in TileLang and records per-case CSV rows.

- Driver: `benchmark/autotune/run_gemm_autotune_e2e.py`
- Kernel source: `examples/gemm/example_gemm_autotune.py`
- Main timing signals:
  - `end_to_end_s`
  - `compilation_s`
  - `benchmark_s`
- TIR equivalence checker: `benchmark/autotune/run_gemm_grouped_tir_equivalence.py`
- Simple dump_ir checker (vector_add): `benchmark/autotune/run_vector_add_grouped_tir_dump.py`

## Current Default Setup

When no arguments are passed, the script runs:

- shape list: only `square_large = 4096x4096x4096`
- `cpu_count`: `16`
- `runs`: `1`
- `warmup`: `3`
- `rep`: `20`
- `timeout`: `180`
- `with_roller`: `False`
- `use_pipeline`: `False`
- `enable_grouped_compile`: `False`
- `group_compile_size`: `2`
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

Enable compile/benchmark pipelining (fallback two-phase mode):

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py --pipeline
```

Enable grouped compile units (current phase: CUDA + cython only, otherwise auto-fallback):

```bash
python benchmark/autotune/run_gemm_autotune_e2e.py \
  --execution-backend cython \
  --grouped-compile \
  --group-compile-size 2
```

## Grouped vs Normal TIR Equivalence

This experiment compares **pre-codegen device TIR** for each GEMM config between:

- normal path lowering (`lower_to_host_device_ir`)
- grouped path using `compile_grouped_unit_tvm_ffi(...)` and checking each returned `jit_kernel.artifact.device_mod`

The key signal is `same_device_tir_before_codegen` in CSV:

- `True`: pre-codegen device TIR is equivalent after canonicalizing grouped suffix names.
- `False`: potential behavior/perf risk; mismatched device TIR scripts are dumped for inspection.

Run:

```bash
python benchmark/autotune/run_gemm_grouped_tir_equivalence.py \
  --shape 4096x4096x4096 \
  --with-roller \
  --topk 20 \
  --max-configs 20 \
  --group-compile-size 4 \
  --csv benchmark/autotune/results/gemm_grouped_tir_equivalence.csv
```

Output files:

- CSV rows: `benchmark/autotune/results/gemm_grouped_tir_equivalence.csv`
- mismatches (if any): `benchmark/autotune/results/gemm_grouped_tir_mismatch/*.tir`

## Manual DumpIR Check (VectorAdd)

This script uses a simple 1D `vector_add` kernel and compares pre-codegen device TIR between:

- normal per-config lowering
- grouped path through `compile_grouped_unit_tvm_ffi(...)`

It also enables `tl.enable_dump_ir` for both paths so you can inspect pass-by-pass IR dumps directly.

Run:

```bash
python benchmark/autotune/run_vector_add_grouped_tir_dump.py \
  --N 4096 \
  --group-compile-size 2 \
  --csv benchmark/autotune/results/vector_add_grouped_tir_dump.csv \
  --dump-root benchmark/autotune/results/vector_add_tir_dump
```

Key outputs:

- CSV summary: `benchmark/autotune/results/vector_add_grouped_tir_dump.csv`
- normal dump_ir tree: `benchmark/autotune/results/vector_add_tir_dump/normal/`
- grouped dump_ir tree: `benchmark/autotune/results/vector_add_tir_dump/grouped/`
- mismatched pre-codegen TIR files: `benchmark/autotune/results/vector_add_tir_dump/mismatch/`
- normal device codegen source per config: `.../normal/cfg_*/device_kernel_source.cu`
- grouped device codegen source per compile unit: `.../grouped/unit_*/device_kernel_source_grouped.cu`

## CSV Columns

- `shape_tag`, `shape_class`, `shape`
- `problem_gflop`
- `cpu_count`, `run_idx`
- `status`, `error`
- `num_configs`
- `end_to_end_s`
- `compilation_s`: compile stage time from futures submission to compile collection
- `benchmark_s`: benchmark stage time spent in the benchmark loop
- `compile_stage_totals_s`: JSON map of summed compile stage timings across compiled configs
- `compile_stage_avg_ms`: JSON map of per-stage average time in ms per config
- `enable_grouped_compile`, `group_compile_size`
- `grouped_compile_active`: whether grouped mode is actually active for this run
- `num_compile_units_submitted`, `avg_group_size`
- `grouped_compile_reason`: fallback reason when grouped mode is requested but inactive
- `best_latency_ms`, `best_tflops`
- `ref_latency_ms`, `ref_tflops`
- `cpu_count_env`
- `execution_backend`, `profile_backend`
- `warmup`, `rep`, `timeout`
- `with_roller`, `topk`, `skip_check`, `cache_input_tensors`, `use_pipeline`
- `best_config` (JSON string)

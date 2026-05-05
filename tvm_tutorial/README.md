# TVM / TileLang Tutorial Split

This tutorial is split into two runnable files:

1. [`01_one_kernel_pipeline.py`](/m-coriander/coriander/ziren/code/tilelang/tvm_tutorial/01_one_kernel_pipeline.py)
2. [`02_multi_kernel_merge.py`](/m-coriander/coriander/ziren/code/tilelang/tvm_tutorial/02_multi_kernel_merge.py)

Shared helpers are in:

- [`tutorial_common.py`](/m-coriander/coriander/ziren/code/tilelang/tvm_tutorial/tutorial_common.py)

## 1) One Kernel Whole Pipeline

Run:

```bash
python tvm_tutorial/01_one_kernel_pipeline.py \
  --target cuda \
  --dump-dir tvm_tutorial/out_one
```

This covers:

- kernel definition (`@T.prim_func`)
- elaboration to `PrimFunc`
- lowering to `host_mod` + `device_mod` under `tvm.transform.PassContext`
- device codegen and host codegen
- artifact dump for every phase

Main artifacts:

- `kernel_add.00_primfunc.tir` (or `kernel_sub.*` if `--kernel-op sub`)
- `*.01_host_lowered.tir`
- `*.02_device_lowered.tir`
- `*.03_lower_stage_seconds.json`
- `*.04_device_codegen.txt`
- `*.05_host_codegen.txt`
- `summary.json`

## 2) Multiple Kernels Merge / Grouped Compile Style

Run:

```bash
python tvm_tutorial/02_multi_kernel_merge.py \
  --target cuda \
  --dump-dir tvm_tutorial/out_grouped
```

This covers:

- lowering two kernels independently
- merging their `device_mod` into one `tvm.IRModule`
- grouped device codegen once
- per-kernel host codegen + `import_module(grouped_device_rt_mod)`

Dispatch trace (optional):

```bash
python tvm_tutorial/02_multi_kernel_merge.py \
  --target cuda \
  --dump-dir tvm_tutorial/out_grouped \
  --dispatch-trace
```

Main artifacts:

- `kernel_a.00_primfunc.tir`, `kernel_b.00_primfunc.tir`
- `grouped.00_merged_device_ir.tir`
- `grouped.01_merged_device_symbols.json`
- `grouped.02_device_codegen.txt`
- `grouped.03_host_codegen_from_kernel_a.txt`
- `grouped.03_host_codegen_from_kernel_b.txt`
- `grouped.dispatch_trace.json` (only with `--dispatch-trace`)
- `summary.json`

## Common Flags

- `--target`
- `--target-host`
- `--n`
- `--pass-config-json`
- `--compile-device`
- `--dispatch-trace` (script 2 only)

`--compile-device` enables real backend compile (e.g. nvcc for CUDA). Without it, the tutorial uses source-only device codegen (`device_codegen_without_compile`) to make internals easier to inspect.

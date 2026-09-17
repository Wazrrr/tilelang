# CUDA portability review

The five families use their existing example builders and the same configuration
pools on Ampere, Hopper and Blackwell. No architecture-specific experiment kernel
or pool is required. The named presets are A100 (`sm_80`), H200 (`sm_90a`) and
B200/GB200 (`sm_100a`); other GPU models/SM versions require an explicit device
manifest, not a new kernel adapter.

The review corrected four framework issues:

- System planning constructed a Hopper device just to enumerate a shared pool.
  It now reads the family pool directly and detects the execution target at runtime.
- Smoke checks demanded WGMMA/TMA on Hopper and TCGEN05/TMA on Blackwell.
  They now record actual emitted instructions and accept the compiler's valid
  matrix lowering. Grouped GEMM is included in matrix-instruction checks.
- A multi-target study bound one GPU before iterating its targets. Each local
  target now selects an idle GPU with matching compute capability/model and
  restores the caller's visibility afterward. The sharded oracle collector also
  respects `CUDA_VISIBLE_DEVICES`.
- Carver advertised Blackwell support despite its unchanged CUDA precision
  dispatch stopping at Hopper. GEMM/attention record `unsupported` there before
  model execution. No Carver source or substitute architecture was introduced.

System optimization variants, numerical references, config IDs, baseline reuse,
and contention monitoring are shared. Baselines remain separate per observed GPU
model. Ampere measurements do not become Blackwell baselines.

## Validation on 2026-09-16

All 50 system/census/TileTune/baseline plans across the five families and three
CUDA presets passed using only the Python standard library. The focused test
run passed 89 tests with five hardware tests skipped. After the final GPU-routing
checks, the complete `testing/python/experiments` suite passed 312 tests with
43 hardware tests skipped. Ruff and whitespace checks passed. All 53 audited
family-kernel/pool/case/reference and Carver source files were unchanged.

Representative configurations for both final cases of all five families were
cross-compiled to PTX and assembled to native cubins for Ampere and Blackwell:
20/20 passed both stages. This includes causal
attention, KDA tails, irregular softmax and ragged grouped GEMM. Each configuration
comes from the unchanged expanded pool. The KDA representative uses `num_stages=0`;
the initial staged Ampere candidate exposed the example's existing overlapping
shared-buffer-write limitation. Other configurations may likewise fail and must
remain recorded outcomes rather than being removed from the pool.

The [validation artifacts](utils/results/cuda-portability-20260916/) contain the
plans, test logs, generated CUDA/PTX, compiler results and source hashes. GPU
execution was disabled during cross-compilation. This host only exposes occupied
H200s, so A100/B200 numerical correctness, profiling, full sweeps and system
speedups remain unverified on hardware. Compilation is not a performance claim.

## Same entry points on another CUDA target

For example, replace `gemm` with any family folder and use `ampere` or `blackwell`:

```bash
python -m experiments.gemm.tiletune.run --suite smoke --device ampere
# Explicit baseline collection, once per GPU/model and fixed pool.
python -m experiments.gemm.tiletune.run --suite full --device ampere --run-baselines
# Subsequent TileTune revisions read the saved baselines.
python -m experiments.gemm.tiletune.run --suite full --device ampere
# System experiments detect the architecture of the selected idle GPUs.
python -m experiments.gemm.system.run --variant all --output experiments/gemm/results/system-a100
```

Multi-GPU system variants require at least two idle GPUs of the same model.
HIP and Ascend remain separate backend integrations; this review covers the
three CUDA architecture presets.

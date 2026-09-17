# GEMM experiments

This experiment uses one kernel: [`make_autotune_kernel_builder`](../../examples/gemm/example_gemm_advanced_autotune.py)
from the advanced GEMM example. `kernel.py` supplies inputs, a numerical reference
and the runner interface. There are no local TileLang kernels or legacy runners.

A has shape (M,K), B has shape (N,K), and the kernel computes C=A@B.T using FP32
accumulation and the example's shared-memory output. All named-suite workloads
use FP16 and explicitly record `transpose_b=True`. BF16 is also supported.
B preparation is outside kernel timing. Batched, NN/TN, fused-epilogue and FP8
requests are unsupported by this experiment.

## One configuration set

[`spaces.py`](spaces.py) defines the only pool, `expanded`, with 2,304 configurations:
exactly eight times the example's 288 configurations. Tile sizes are sampled more
finely; the other parameters keep the example's ranges.

| Parameter | Values |
| --- | --- |
| `block_M` | 32, 64, 96, 128, 192, 256 |
| `block_N` | 32, 64, 96, 128, 192, 256 |
| `block_K` | 16, 32, 48, 64 |
| `num_stages` | 0, 1, 2, 3 |
| `thread_num` | 128, 256 |
| `enable_rasteration` | True, False |

The example fixes warp policy to Square and enabled rasterization to panel 10.
The complete original grid and its measured Hopper winner (128x256x64, 3 stages,
256 threads, rasterization enabled) are included. Every method uses the same
configuration dictionaries and order. There is no 1,024-config cap, protected
subset or `current`/`large`/`exhaustive` GEMM preset. Smoke/development budgets
select recorded indices from this pool. Explicit CUDA/HIP configs must also be
members of it.

The pool is identical across target devices. All declared candidates are
attempted; compilation and correctness failures are recorded. The 2,304 count
does not assert that every candidate compiles or yields distinct device code.

## Files and cases

- `cases.py`: train/validation/development/final workload definitions.
- `spaces.py`: the sole compiler-free pool generator and workload contract.
- `kernel.py`, `reference.py`: example adapter and independent numerical check.
- `tiletune/run.py`, `census.py`: common study and census entry points.
- `system/run.py`: compiler/benchmark scheduling comparisons on the same pool.
- `carver.py`, `service_audit.py`: comparison model and diagnostics using that pool.
- `heuristics/GPU/`: results from completed sweeps and winner validation.

| Split | M,N,K |
| --- | --- |
| Training A | 32,4096,4096 |
| Training B | 512,14336,4096 |
| Validation | 2048,4096,4096 |
| Development decode | 64,4096,4096 |
| Development prefill | 512,4096,4096 |
| Development FFN down | 512,4096,14336 |
| Development large projection | 2048,4096,4096 |
| Development FFN up | 2048,14336,4096 |
| Final decode | 128,4096,4096 |
| Final prefill | 1024,4096,4096 |
| Final FFN down | 1024,4096,14336 |
| Final large projection | 4096,4096,4096 |
| Final FFN up | 4096,14336,4096 |

These represent decode and prefill hidden projections plus both FFN directions
for a 4096-wide decoder. The stable `gemm_square*` workload identifiers are
retained for the two large-prefill cases.

## Commands

```bash
# Inspect the entire final pool, without importing the compiler or using a GPU.
python -m experiments.gemm.tiletune.run --suite final --device hopper --plan

# Time complete final sweeps on all idle H200s, with contention monitoring.
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m experiments.common.brute_force \
  --workloads gemm_square gemm_square_large --workers 8 \
  --output experiments/results/h200-gemm-expanded

# Compile/check a smoke subset, retaining original pool indices.
python -m experiments.gemm.tiletune.run --suite smoke --device hopper \
  --output experiments/gemm/results/smoke

# Compare TileTune, Carver, XGBoost and brute force on development shapes.
python -m experiments.gemm.tiletune.run --suite development --device hopper \
  --output experiments/gemm/results/development
```

The monitored sweep disables compilation/autotune caches and checks each
successful candidate against the numerical reference. CUDA-event timing uses
10 ms warmup, 50 ms repetition and a 256 MiB cache flush outside the measured
interval. The minimum correct measured latency selects the winner; seven fresh
measurements validate it. Foreign GPU processes invalidate an affected shard,
which is retried. Monitoring polls every second and cannot exclude shorter
interference.

Space version 4 identifies this pool. Old measured records are archived under
`experiments/results/gemm-pre-single-pool-20260916/heuristics/`; their timings
belong to their original programs and pools. New results include source/build
hashes, raw outcomes, device observations and elapsed time.

The completed [H200 sweep](heuristics/H200/README.md) predates the serving-shape
update and retains its original 4096³/8192³ identity. It is historical evidence,
not a measurement of the current final workloads.

## System ablations and reusable baselines

```bash
python -m experiments.gemm.system.run --variant all --plan
python -m experiments.gemm.system.run --variant all \
  --output experiments/gemm/results/system-v1
python -m experiments.gemm.tiletune.run --suite full --device hopper --run-baselines
python -m experiments.gemm.tiletune.run --suite full --device hopper \
  --output experiments/gemm/results/tiletune-revision-a
```

System runs support baseline, pipeline, grouped, multi_gpu and combined modes
on all five final FP16 cases. New TileTune output directories reuse verified baseline
bundles while the kernels, pools and measurement environment remain compatible.
Baselines live under `results/<GPU model>/baselines/`, with `current.json` pointing
to the saved bundle. Only `--run-baselines` collects or refreshes them; ordinary
TileTune runs are read-only and require an existing compatible bundle.
Baseline XGBoost uses a fixed seed independently of TileTune repeats. Carver uses the original GEMM policy. See the [workflow guide](../README.md)
for GPU monitoring, baseline identity, artifact paths and arbitrary-K comparisons.

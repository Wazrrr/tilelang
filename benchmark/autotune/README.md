# Reproducing New Carver experiments

Run commands from a checkout of this commit. The portable entry point is
`python -m benchmark.autotune.validate_new_carver_generalization`. It uses actual
GEMM/attention PrimFuncs, fixed device probes and the full original config grids.
No candidate timing fits the model and no top-K pruning is applied.

## Build on the destination node

Use an environment with a CUDA-enabled PyTorch compatible with the node's driver
and CUDA toolkit. Follow the repository's normal source-build dependencies:

```bash
git submodule update --init --recursive
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-build-isolation
python -m pip install scipy
```

For an existing development build, rebuild the native adapter and compiler fix:

```bash
cmake -S . -B build
cmake --build build -j8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Select an idle GPU using `nvidia-smi`, or use the devices assigned by your job
scheduler. `CUDA_VISIBLE_DEVICES` controls execution. The optional `--gpu` flag
only labels the physical device in snapshots; omitting it queries the UUID of
the current visible device. The commands below use one GPU visible as device 0.
No local conda wrapper, historical result directory, or H200 profile is required.

## Profile, freeze, run, evaluate

On the A100 node, generate a new device profile. This measures fixed primitives
once for FP16 and BF16, then reuses the common rates. On Hopper the same command
also measures FP8 WGMMA. It does not benchmark any candidate kernel.

```bash
export CUDA_VISIBLE_DEVICES=0
python -m benchmark.autotune.validate_new_carver_generalization profile \
  --profile /tmp/carver-a100-device.json
python -m benchmark.autotune.validate_new_carver_generalization freeze \
  --profile /tmp/carver-a100-device.json --root /tmp/carver-a100-all --split all
python -m benchmark.autotune.validate_new_carver_generalization worker \
  --root /tmp/carver-a100-all
python -m benchmark.autotune.validate_new_carver_generalization evaluate \
  --root /tmp/carver-a100-all
```

Use persistent storage instead of `/tmp` for results you want to retain. A profile
fingerprint mismatch requires a new file or an explicit `profile --refresh`.
Refresh before freezing; frozen experiments reject modified profiles or sources.
The freeze records the target/device limits, source/profile hashes, native build,
git commit/status, workload declarations, settings and measurement protocol.

Choose a split to control workload scope; grids are always complete:

| Split | Workloads | Configs per case |
| --- | --- | --- |
| `baseline` | 4096³ FP16 GEMM; B1 H16 S4096 D128 attention, noncausal and causal | 288 GEMM / 128 attention |
| `development` | Square, tall, short-K and transposed FP16 GEMMs | 288 |
| `holdout` | Rectangular FP16, BF16 TN, FP8 NN; D64, D256 and unequal-Q/K attention in both modes | 288 GEMM / 128 attention |
| `all` (default) | All 17 cases above | 3,616 config outcomes |

Each case runs in a fresh process, with JIT/autotuner caches disabled, four CPU
compilation workers, five warmup and thirty measurement repetitions, CUDA graph
benchmarking and seed 123. Input/reference checks cover every benchmarked config.
Invalid configurations still receive outcomes. The default is report-only:
GEMM uses zero spill/local tolerance; attention records spills/local memory with
no byte cap and allows 32 modeled registers per consumer above its demand budget.
Physical SM capacity stays strict. Both pressure decisions remain in the report.

For only the original 4096³ and attention experiment, use `--split baseline` when
freezing. For a single case from a frozen experiment:

```bash
python -m benchmark.autotune.validate_new_carver_generalization run \
  --root /tmp/carver-a100-all --case gemm_4096
```

Workers skip completed cases, making interrupted suites resumable. A failed case
returns a nonzero exit status and preserves its logs and summary. Do not overwrite
a completed experiment; use a fresh root for new measurements.

## Two GPUs and mode comparisons

Freeze once on one device. Workers may share that root on identical device types
with the same build and software fingerprint. Shards partition workloads, never
a workload's config grid. Launch these commands concurrently in separate shells
or jobs, after checking both devices are idle:

```bash
CUDA_VISIBLE_DEVICES=1 python -m benchmark.autotune.validate_new_carver_generalization worker \
  --root /tmp/carver-a100-all --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=2 python -m benchmark.autotune.validate_new_carver_generalization worker \
  --root /tmp/carver-a100-all --shard 1 --shards 2
```

For disabled/report-only/rejection comparisons, freeze separate roots with
`--mode disabled`, `--mode report_only`, or `--mode reject`, reusing the same
profile. Run each root on an idle device. Add `--group-size 4` to `freeze` to test
grouped compilation; the default is individual compilation. Rejection can reduce
the measured set, so use report-only results to evaluate pruning accuracy.

## Outputs and interpretation

- `freeze.json`, `device.json`, `sources/`: immutable experiment inputs.
- `<case>/summary.json`, `carver.json`: every original config, pressure evidence,
  compiler resources, correctness/benchmark status, measured winner and latency.
- `<case>/config_<index>.cu`, `compiled_<index>.json`, `result_<index>.json`:
  generated code and per-config checkpoints.
- `<case>/timings.tsv`, `benchmarks.tsv`, and per-case logs: orchestration stages,
  measured latency and failures.
- `evaluation.json`: numeric-score coverage, winner rank/tie interval, Spearman
  correlation, top-K retained performance, rejection effects, summed stage-cost
  percentages and separate end-to-end wall time. Both `pipeline_time` and
  `traffic_waves` are evaluated. Unscored positions never count as top-K predictions.

For predictions without measuring candidates, run `predict --root ... --profile
... --destination ...`. It uses the frozen device limits and workload declarations,
with the current analyzer and the supplied profile, and does not need CUDA access.

**A100 coverage:** pressure, tile traffic, waves, compilation and measurements
cover FP16/BF16 grids. Stage-zero MMA GEMM and supported attention reductions have
timing estimates. Positive-stage Ampere software-pipeline scheduling is currently
unmodeled: its `pipeline_time` score remains unknown, without dropping the config.
Report this coverage alongside ranking accuracy. The runner records all 288 FP8
configs as unsupported on A100, which lacks FP8 tensor-core instructions. Hopper
instruction/profile mismatches also stay unscored. Do not transfer H200 profile
rates to an A100 or interpret cross-compilation as A100 performance validation.

## Tests and individual examples

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest testing/python/new_carver/ -q
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  testing/python/transform/test_tilelang_transform_ws_phase_lag.py -q
```

The phase-lag test requires Hopper and skips on A100. The portability tests include
CPU analysis with explicit A100 limits and sm_80 device cross-compilation.

`benchmark_new_carver_gemm.py`, `benchmark_new_carver_attention.py`, and
`benchmark_new_carver_fp8.py` provide individual-case comparisons, including
selected-index debugging. The FP8 script/example requires Hopper. See
[the model documentation](../../docs/new_carver.md) for API use and formulas, and
[the preceding H200 findings](../../docs/new_carver_validation.md) for known model
weaknesses. Historical fitted-profile scripts and raw experiment artifacts are
not dependencies of this workflow.

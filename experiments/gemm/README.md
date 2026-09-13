# GEMM experiments

Run all commands from the repository root in an environment with TileLang,
PyTorch, and CUDA. Each command creates a timestamped run subdirectory under
`--output`, so the same command can be repeated to retain multiple versions.
Add `--run-name v2` for a named version; an existing name is rejected.

Both experiment types use [kernel.py](kernel.py): the advanced-autotune
`A @ B.T` GEMM, with FP32 accumulation, a shared-memory epilogue, and the same
288 configurations. The system experiments default to BF16; TileTune supports
both FP16 and BF16.

## System experiments: five cases

[system/run.py](system/run.py) measures autotuning wall time and the measured
winner's kernel latency. TileTune and legacy filtering are disabled.

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

## Cost-model comparison: brute force, Carver, TileTune

[tiletune/run.py](tiletune/run.py) controls all methods through `--method`.
The default is `tiletune` with `--top-k 20`. `all` runs the three implemented
methods in separate processes and prints a comparison.

| Method | Candidate selection | Compilation and benchmark budget |
| --- | --- | --- |
| `brute_force` | All 288 configurations, no cost-model filtering | Entire grid |
| `carver` | Legacy Carver cost on the same 288-config grid | At most K candidates |
| `tiletune` | TileTune `pipeline_time` cost on the same grid | At most K candidates |

Brute force means exhaustive autotuning. The example's `--use_autotune` disabled
path instead runs one heuristic configuration; that is not this baseline.

### Run the comparison

```bash
# Full FP16 comparison: the same K=20 for Carver and TileTune.
CUDA_VISIBLE_DEVICES=2 python -m experiments.gemm.tiletune.run \
    --method all --top-k 20 \
    --m 4096 --n 4096 --k 4096 --dtype float16 \
    --device-profile experiments/profiles/h200-tiletune-topk.json \
    --output experiments/results/gemm/comparison_fp16

# Same experiment in BF16.
CUDA_VISIBLE_DEVICES=2 python -m experiments.gemm.tiletune.run \
    --method all --top-k 20 \
    --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
    --device-profile experiments/profiles/h200-tiletune-topk.json \
    --output experiments/results/gemm/comparison_bf16
```

Run the methods separately with the same arguments:

```bash
# 1. Exhaustive baseline; --top-k does not limit brute force.
CUDA_VISIBLE_DEVICES=2 python -m experiments.gemm.tiletune.run \
    --method brute_force --m 4096 --n 4096 --k 4096 --dtype float16 \
    --output experiments/results/gemm/brute_force

# 2. Legacy Carver on the common grid.
CUDA_VISIBLE_DEVICES=2 python -m experiments.gemm.tiletune.run \
    --method carver --top-k 20 --m 4096 --n 4096 --k 4096 --dtype float16 \
    --output experiments/results/gemm/carver

# 3. TileTune on the common grid.
CUDA_VISIBLE_DEVICES=2 python -m experiments.gemm.tiletune.run \
    --method tiletune --top-k 20 --m 4096 --n 4096 --k 4096 --dtype float16 \
    --device-profile experiments/profiles/h200-tiletune-topk.json \
    --output experiments/results/gemm/tiletune
```

### Selection and measurement contracts

Both models rank the supplied grid before compiling any selected configuration.
TileTune elaborates and analyzes each candidate once, retains selected PrimFuncs,
and compiles only its top K finite, eligible scores. Unknown and pressure-rejected
candidates are excluded. Both methods break equal scores by original grid order
and report tie intervals. A compilation or benchmark failure consumes one of the
selected attempts; no extra candidate replaces it. If fewer than K candidates
are eligible, the reports show the shortfall. No eligible selection is a failed
experiment, not a successful zero-cost result.

### Original Carver baseline

The entire `tilelang/carver/` package and its original tests are restored to
upstream commit `3e4a05443b3bcebf0339c1b9b0a460bbf733353c` (2026-07-30),
before the local autotune/TileTune experiments. Carver was introduced in
`cd191889b` on 2025-02-10; the CUDA warp-tile score fix dates to 2025-08-15
(`2bd2d69e6`). This snapshot retains upstream compatibility changes, including
the current TVM `tirx` API, with the original CUDA policy and resource model.
The local WGMMA/WS model extensions, register estimates, hint metadata,
architecture parser changes, and added CUDA-driver helpers have been removed.
Local Carver hint-conversion edits in the legacy examples, benchmarks, and
autotuning tutorial have also been reverted.

Each run records SHA-256 hashes of the current Carver, TileTune, autotuner, and
experiment Python sources in `experiment.json`. These identify the code used
for that run; the upstream restoration commit above provides historical context.
The adapter records both target spellings in `carver.json`. TileTune's extra
device-limit query lives in TileTune and uses the original generic driver query,
leaving Carver unchanged.

The old parser accepts `sm_90` but cannot parse `sm_90a`. The experiment adapter
passes `sm_90` to the Carver model on Hopper; **all methods still compile the
same GEMM kernel for the actual `sm_90a` target**. This spelling conversion lives
outside Carver. It adds no WGMMA/WS modeling or revised resource limits.

The Carver adapter in [carver.py](carver.py) calls the restored upstream
`TensorCorePolicy` on each supplied output tile and reduction step. Its score is
`(traffic_bytes + 1) * num_wave`, retaining its own shared-memory/register and
thread-feasibility checks. Stages 0 and 1 both mean one shared-memory
copy to this model. Thread count affects feasibility, while the ranking keeps
Carver's occupancy estimate. Rasterization is not distinguished by the score.
These limitations appear as ties; the adapter adds no TileTune model terms.
The adapter also retains Carver's resource limits: `smem_cap` is the device's
non-opt-in per-block shared-memory limit, and `max_smem_usage` is twice that
value. On the tested H200 these are 48 KiB and 96 KiB, whereas the device permits
227 KiB per block with opt-in and has 228 KiB per SM. This can reject configurations
that TileLang successfully compiles. Selection quality therefore includes both
resource feasibility and ranking; isolating the ranking formulas would require
a separate experiment with matched resource limits.
This is a **cost-model comparison on a common grid**, not Carver's native
candidate-generation search.

All methods compile the same [kernel.py](kernel.py), use the same seeded inputs,
and check results against PyTorch. Defaults are one visible GPU, CUDA-graph
benchmarking, 4 compilation workers, and `--warmup 10 --rep 100`. Both TileLang
kernel and autotune caches are disabled before imports. `all` runs Carver and
TileTune before brute force; neither model reads the exhaustive measurements.
Pipeline overlap and multi-GPU benchmarking are disabled. Grouped compilation is
off by default; `--group-size 2` enables it equally for all three methods.

TileTune measures or loads primitive rates before tuning. The profile is frozen
before candidate measurements, reused across compatible runs, and its preparation
time is reported separately. Use a new profile path after an incompatible
native build or device change. The profile supports A100 and Hopper. Streaming
memory rates are the default model input; `--memory-regime cached` selects the
cached-memory rates.

### Results

```text
<output>/<UTC timestamp>/
├── carver/              experiment.json, carver.json, outcomes.json, summary.json, TSVs
├── tiletune/            experiment.json, tiletune.json, outcomes.json, summary.json, TSVs
├── brute_force/         experiment.json, outcomes.json, summary.json, TSVs
├── carver.log
├── tiletune.log
├── brute_force.log
├── validation_timings.tsv
└── comparison.json
```

Each model report preserves all configurations, original indices, scores, ties,
selection decisions, and failures. Unselected analyzed configurations are marked
`not_selected`. `summary.json` reports the method's fastest successful selected
candidate, latency, TFLOPS, predicted rank, selected/successful/failed counts,
profile preparation time, selection time, and tuning wall time. `timings.tsv`
contains stage measurements. `compile_work_seconds` sums compilation-worker
work; `benchmark_work_seconds` includes checking and measurement. They are work
totals, so they should not be added to infer parallel wall time.

`all` then recompiles only the already chosen winners and remeasures them in
shuffled order with the same inputs. It reports the median of 5 measurements
(`--validation-repeats`) and stores every sample. This validation work is timed
separately and never changes the selected configurations.

The comparison includes:

- Validated winner latency/TFLOPS and speed relative to the brute-force winner.
- Tuning speedup, excluding device-profile preparation and final validation.
- `top_k_oracle_retained_performance` (the table's `Oracle@K`): exhaustive best
  latency divided by the best exhaustive latency within the frozen top-k set.
  This evaluates selection quality using a common measurement table. Candidates
  without successful exhaustive measurements are excluded and coverage is reported.
- The selected winner's predicted rank, and the brute-force winner's rank under
  each model, with ties and whether it was selected.

These two performance comparisons use different measurements: final validation
measures the chosen winners again, while `Oracle@K` evaluates the full selected
set in the exhaustive table. A ratio near 100% means the selected set retains
near-best grid performance. Results can show either model winning; one shape is
insufficient to establish general cost-model superiority.

### Rerun with original Carver (H200, 2026-09-08)

Fresh comparisons use the pinned original baseline, `M=N=K=4096`, K=20,
4 compilation workers, `--warmup 10 --rep 100`, CUDA graphs, and 5 final
winner measurements. Device-profile preparation and final validation are
excluded from tuning wall time. The runner does not control GPU clocks.

| Dtype | Method | Validated winner (ms) | Tuning (s) | Oracle@K | Selected winner's model rank |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | Brute force | 0.229096 | 671.482 | 100.00% | — |
| FP16 | Original Carver | 0.442541 | 47.639 | 48.72% | 4 |
| FP16 | TileTune | 0.227078 | 40.881 | 100.00% | 16 |
| BF16 | Brute force | 0.217954 | 673.327 | 100.00% | — |
| BF16 | Original Carver | 0.426048 | 48.747 | 48.04% | 4 |
| BF16 | TileTune | 0.211586 | 41.763 | 100.00% | 11 |

FP16 original Carver selected configuration #131 (rank 4, tied ranks 1–8).
TileTune selected #255 (rank 16, tied ranks 9–16). The exhaustive winner #190
is also in TileTune's top-20, at rank 11 (tied 9–16), and is rejected by Carver's
original resource model. Final remeasurement can reverse close winner
orderings; it does not establish an advantage over exhaustive search.

BF16 original Carver also selected #131 (rank 4, tied 1–8); TileTune selected
#190 (rank 11, tied 9–16). The exhaustive winner #255 is in TileTune's top-20
at rank 16 (tied 9–16), and is rejected by Carver. In both dtypes the restored
Carver top-20 has zero overlap with the superseded experimental Carver top-20.

All 288 exhaustive candidates and all 20 candidates selected by each model
compiled and passed correctness checks in each dtype run. NVCC ran 288/20/20
times per run, confirming cold compilation for brute force/Carver/TileTune.
TileTune's FP16/BF16 tuning wall time was 16.43×/16.12× shorter than brute
force, including 9.051/8.989 s of analysis and selection.

FP16 data:
`experiments/results/gemm/comparison_fp16/20260908T102804.496571Z/`.
BF16 data:
`experiments/results/gemm/comparison_bf16/20260908T104212.048275Z/`.
The result files preserve the full grid, candidate outcomes, source hashes,
baseline identity, profile, selected winner ranks, and exhaustive winner ranks.
All six method runs recorded identical source hashes; both TileTune reports use
analysis version 17 and the same previously prepared device-profile file.

### Superseded experimental runs

The earlier FP16 run `20260908T095253.742964Z` and BF16 run
`20260908T100632.642520Z` used the locally modified WGMMA Carver model.
They are **not results for the original Carver baseline**. Their data remains
in the timestamped `comparison_fp16`/`comparison_bf16` output folders for
provenance. Restoring upstream Carver changed its top-20 set for both dtypes;
fresh comparisons use the pinned baseline above.

### Run sizes and validation

`--config-indices 0 8 16 24 --top-k 2 --m 256 --n 256 --k 256` gives a short
four-config run. Subset indices, rankings, and the exhaustive winner refer only
to that supplied subset. All methods accept `--workers`, `--warmup`, `--rep`,
`--timeout`, `--seed`, and `--backend`. Reuse `--output` to preserve timestamped
versions, or choose a new explicit `--run-name`.

For broader evaluation, keep K fixed before measuring and use predefined square,
tall, and wide GEMMs with FP16/BF16. Run GPU experiments sequentially to avoid
competing benchmarks. The focused tests are:

```bash
python -m pytest testing/python/tiletune/test_top_k.py testing/python/experiments/ -q
```

See the [shared experiment guide](../README.md) for the FP8/attention experiment
contracts and the [TileTune guide](../../docs/tiletune.md) for the library API.

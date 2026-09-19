# Archived softmax measurements

Softmax was removed from the active experiment suite on 2026-09-17 and replaced by `../gemm_fp8/`. Historical results and heuristic provenance remain here; executable source was archived under `../results/archive/softmax-source-20260917/`.

## Direct PrimFunc analysis check

[analyze.py](analyze.py) defines a standalone, stable row softmax with FP16
input/output and FP32 intermediate tiles. It calls `analyze_prim_func` directly
with `ranking_metric="memory", memory_diagnostics=True`, a CUDA SM90a target, and a supplied SM count of
132. It adds no TileTune family recognizer, cost formula, fragment layout hint,
compute profile, or changes to the analyzer. The configuration pool is explicitly
supplied: rows per CTA in `{1, 2, 4, 8}` and threads in `{128, 256}`.

```bash
python -m experiments.softmax.analyze \
  --output experiments/results/h200-unified-memory/softmax-generality.json
```

The saved original report scored all 24 configurations with analysis version 34.
The current runner opts into diagnostics to preserve its dependency and liveness
inspection; version 35 disables those reports by default in memory mode.
The runner replaces family dispatch, family construction, pipeline timing,
occupancy, and warp-specialization prediction with functions that raise if called.
No model implementation was changed for this check.

| Shape | Scored and eligible | Primary score groups | Selected at strict alpha=0.5 |
|---|---:|---:|---:|
| 256 × 128 | 8/8 | 3 | 4/8 |
| 257 × 1000, masked rows/columns | 8/8 | 3 | 4/8 |
| 4096 × 4096 | 8/8 | 1 | 0/8 |

For each configuration, the analyzer captures two reductions and three scalar
operation groups, their memory accesses, eight dependency edges, and live tile
storage. The specialization is `generic`; no memory effects remain unknown.
The original PrimFunc is unchanged. The full reports are saved in
[softmax-generality.json](../results/h200-unified-memory/softmax-generality.json).

This checks CPU analysis coverage only. It performs no compiler lowering, GPU
execution, numerical correctness comparison, or oracle measurement. A finite
score does not prove compilation success or good ranking: all eight large-shape
configurations tie, and changing only the thread count never changes this memory
score. Under the conservative tie rule, all eight large-shape candidates have
rank 8/8. The exact shared score is `9007233614544894`, encoding 524,288
byte-waves, 262,144 access-waves, and pipeline depth 1. An expanding top-K cutoff
inside that group retains all eight; strict alpha=0.5 selects none. The existing
25-case oracle result therefore does not establish a general 50% pruning guarantee.
With diagnostics enabled, dependencies and liveness remain reported facts;
the simplified score does not estimate dependency critical paths, reduction
service time, or thread-dependent occupancy.

A separate pointwise regression adds an exponential, an addition, and an extra
live fragment without changing global accesses. The operation count and reported
live-register peak increase, but the memory score stays identical. This checks
that computation and live storage are reported facts rather than hidden score
terms. Regression tests also compare lean and diagnostic modes for identical
scores and resource rejections.

A separate CPU check of the 256 × 128 case (one row per CTA, 128 threads) with
the older `pipeline_time` mode and the saved H200 primitive profile returned an
unknown score. Both reductions report `no known fragment producer for reduction`.
The graph is captured, but the timing model cannot establish reduction ownership
from this ordinary non-GEMM-produced fragment. See
[the legacy report](../results/memory-ranking/softmax-legacy-analysis.json).

## Generality and the Carver comparison

The direct analyzer supports new compositions of its supported scalar/tile
operations. New opaque operators, unresolved data-dependent accesses, or
unsupported aliasing can require operator metadata or explicit runtime input
values; arbitrary PrimFuncs are not guaranteed to be scoreable. The caller also
supplies the candidate configurations and hardware information.

The memory path bypasses family recognition. The older timing path automatically
recognizes GEMM and attention, but that recognition selects hand-written family
policies (including warp-specialization rules and an optional attention spill
allowance). Automatic selection does not make these policies kernel-agnostic.

Both TileTune and Carver use analytical heuristics. Our fixed-pool Carver
experiment adapters contain explicit family formulas. Carver itself also offers
PrimFunc analysis and implements traffic, resource, and shared-memory-lifetime
heuristics; those capabilities are not exclusive to TileTune. TileTune's useful
distinction here is its capture of the actual supplied TileLang kernel, including
its explicit tile operations, loops, dependencies, and live storage. The new
memory score is deliberately simple and close to traffic-times-waves ranking;
it is not a learned predictor or a calibrated latency estimate.

Relevant source:

- [TileTune mode dispatch](../../tilelang/tiletune/engine.py)
- [Family policies](../../tilelang/tiletune/families/attention.py)
- [Memory score](../../tiletune_core/memory.py)
- [Fixed-pool Carver adapters](../common/carver.py)
- [Carver PrimFunc entry point](../../tilelang/carver/utils.py)
- [Carver resource and lifetime analysis](../../tilelang/carver/roller/policy/default.py)

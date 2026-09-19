# Archived softmax measurements

Softmax was removed from the active experiment suite on 2026-09-17 and replaced by `../gemm_fp8/`. Historical results and heuristic provenance remain here; executable source was archived under `../results/archive/softmax-source-20260917/`.

## Direct PrimFunc analysis check

[analyze.py](analyze.py) defines a standalone, stable row softmax with FP16
input/output and FP32 intermediate tiles. It calls `analyze_prim_func` directly
with `ranking_metric="memory"`, a CUDA SM90a target, and a supplied SM count of
132. It adds no TileTune family recognizer, cost formula, fragment layout hint,
compute profile, or changes to the analyzer. The configuration pool is explicitly
supplied: rows per CTA in `{1, 2, 4, 8}` and threads in `{128, 256}`.

```bash
python -m experiments.softmax.analyze \
  --output experiments/results/memory-ranking/softmax-analysis.json
```

All 24 configurations passed the analysis assertions:

| Shape | Scored and eligible | Distinct (byte score, event score) pairs |
|---|---:|---:|
| 256 × 128 | 8/8 | 3 |
| 257 × 1000, masked rows/columns | 8/8 | 3 |
| 4096 × 4096 | 8/8 | 1 |

For each configuration, the analyzer captures two reductions and three scalar
operation groups, their memory accesses, eight dependency edges, and live tile
storage. The specialization is `generic`; no memory effects remain unknown.
The original PrimFunc is unchanged. The full reports are saved in
[softmax-analysis.json](../results/memory-ranking/softmax-analysis.json).

This checks CPU analysis coverage only. It performs no compiler lowering, GPU
execution, numerical correctness comparison, or oracle measurement. A finite
score does not prove compilation success or good ranking: all eight large-shape
configurations tie, and changing only the thread count never changes this memory
score. Dependencies and liveness remain reported facts/resource-policy inputs;
the simplified score does not estimate dependency critical paths, reduction
service time, or thread-dependent occupancy.

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

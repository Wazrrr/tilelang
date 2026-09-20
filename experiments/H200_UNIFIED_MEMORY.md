# Unified TileTune memory ranking on H200

**All 25 oracle winners have finite scores and conservative tail ranks within the first 50% of their full pools.** Fresh CPU analysis of all **29,200 PrimFuncs** exactly matched the frozen-fact replay. The worst oracle rank is **90/192 (46.875%)**, for `grouped_gemm_decode`. Strict `alpha=0.5` selection retained exactly half of each pool, without splitting ties.

The source is the completed `h200-25shape-20260917T080241Z` FP16/E4M3 study. Kernels, inputs, configuration pools, pre-lowering resource policies, and exhaustive oracle measurements are unchanged. This result does not apply automatically to the newer BF16/block-scaled contract. The ranking validations through version 36 used CPU analysis only; version 37 additionally compiles the oracle configs to validate the post-compile policy below.

An additional [standalone softmax check](softmax/README.md) scores 24/24
PrimFuncs with family helpers disabled. Its 4096×4096 case has eight equal-score
configs, each with tail rank 8/8, so strict alpha=0.5 selects none. Analysis
coverage generalizes to this kernel; the 50% oracle-ranking result does not.

## Shared analysis and selection

The work was made in `dev-h200-new`, using `origin/dev-b200-tiletune` commit `d9b00f4fb33549714b31c007c35315da88938bfd` as the reference for pipeline-depth extraction and strict selection. H200 kernels and pools were preserved.

The scorer receives only resolved memory accesses, launch size, target SM count, and pipeline depth extracted from the IR. It uses no kernel names, family recognition, measured latencies, oracle labels, fitted family weights, or semantic workload adapters. The existing generic integer-input contract supplies immutable metadata values when needed.

```text
waves = ceil(grid_blocks / SM_count)
B = sum(access.bytes * access.visits) * waves
E = sum(access.visits for nonempty accesses) * waves
D = IR pipeline depth (at least 1)
primary order = (B, E, -D)
score = (B * (B + 1) // 2 + E) * 65536 + (65535 - D)
rank = final position of the equal-primary-score group
alpha budget = floor(alpha * original_pool_size)
selection = complete eligible score groups whose tail rank <= budget
```

The triangular encoding is exact because `0 <= E <= B`; adjacent byte-work bands never overlap. The depth radix exceeds the supported depth range. Python integer arithmetic preserves distinct groups above floating-point precision. The score is an ordinal preference, not predicted time or measured DRAM traffic. Smaller byte work is preferred, then fewer logical memory requests, then deeper buffering.

The original byte-only model reached 20/25 within 50%. Porting pipeline-depth preference alone reached 21/25: the four longer/noncausal attention oracles still ended at 168/320. Including request count in the primary order places them at 136/320. This refinement was developed and evaluated on this known fixed pool; no held-out generalization claim is made.

Equal triples remain inseparable. Original index is display order only. Alpha excludes a complete group crossing its cutoff; failures and unknowns remain in the original denominator, and failed selected candidates are never replaced. Explicit resource policies still apply.

## Every oracle score

Exact scores, components, configs, source hashes, selections, and ranks are in [oracle-scores.json](results/h200-unified-memory/oracle-scores.json). The full [fresh IR report](results/h200-unified-memory/live/summary.json) and [frozen replay](results/h200-unified-memory/replay/summary.json) retain their original provenance.

| Case | Oracle score (exact integer) | Tail rank / full pool | Pool share |
|---|---:|---:|---:|
| attention_causal | 9808840555366514686 | 136/320 | 42.500% |
| attention_long_causal | 152221669638575030270 | 136/320 | 42.500% |
| attention_medium_causal | 650770292188708862 | 136/320 | 42.500% |
| attention_noncausal | 9808840555366514686 | 136/320 | 42.500% |
| attention_short_causal | 3518447947022334 | 16/320 | 5.000% |
| gemm_decode | 36593980632334332 | 4/2304 | 0.174% |
| gemm_ffn_down | 4019603767870750716 | 8/2304 | 0.347% |
| gemm_fp8_decode | 9148503817322492 | 4/2304 | 0.174% |
| gemm_fp8_ffn_down | 1013840063050416125 | 160/2304 | 6.944% |
| gemm_fp8_prefill | 85775703883644925 | 160/2304 | 6.944% |
| gemm_fp8_square | 1372410625876885501 | 352/2304 | 15.278% |
| gemm_fp8_square_large | 16812028311231463421 | 368/2304 | 15.972% |
| gemm_prefill | 337910814776360956 | 8/2304 | 0.347% |
| gemm_square | 5406571773598957564 | 200/2304 | 8.681% |
| gemm_square_large | 66230500543356141564 | 264/2304 | 11.458% |
| grouped_gemm_aligned | 251230385063460860 | 2/192 | 1.042% |
| grouped_gemm_decode | 251231110920994812 | 90/192 | 46.875% |
| grouped_gemm_down_aligned | 236589669478498300 | 58/192 | 30.208% |
| grouped_gemm_prefill | 111329478499172348 | 2/192 | 1.042% |
| grouped_gemm_ragged | 339657306092929020 | 2/192 | 1.042% |
| kda_chunk_batched | 1963782996485668862 | 10/720 | 1.389% |
| kda_chunk_long | 7730941636165697534 | 10/720 | 1.389% |
| kda_chunk_medium | 506655086940848126 | 10/720 | 1.389% |
| kda_chunk_regular | 1963782996485668862 | 10/720 | 1.389% |
| kda_chunk_short | 126663803950661630 | 10/720 | 1.389% |

## Use and reproduce

```python
tuner.set_tiletune_args(True, ranking_metric="memory", alpha=0.5)
```

Memory mode now defaults to lean analysis. Set `memory_diagnostics=True` to
restore reaching dependencies, backward-propagation reports, register liveness,
and shared-memory lifetime estimates. These diagnostic analyses do not affect
the score or selection. Launch-limit checks and any backward demands needed to
prove a strict register-policy rejection remain enabled. Post-compile resource
checks remain available. Version 37 enables rejection in the H200 memory
experiment runner with the explicit post-compile policy described below.
Timing modes retain their existing analyses.

The regular runner accepts `--method top_k --metric memory --alpha 0.5`. Alpha and `top_k` are mutually exclusive in `TileTuneConfig`; the CLI alpha option overrides its default top-K budget. `strict_top_k=True` provides the same boundary policy for an explicit integer K. The original reports above use analysis version 34 and `memory.v2`. Lean analysis introduced version 35 and `memory.v3`; a `null` dependency list means collection was disabled. Version 36 defers unnecessary metadata simplification. Current analysis version 37 adds a separate compiler-resource policy, retaining `memory.v3`. The score formula is unchanged; diagnostics and compiler policy are part of the cache key.

```bash
python -m experiments.replay_memory \
  --study /path/to/h200-25shape-20260917T080241Z \
  --output /tmp/h200-memory-replay --alpha 0.5

python -m experiments.validate_memory \
  --study /path/to/h200-25shape-20260917T080241Z \
  --replay /tmp/h200-memory-replay \
  --output /tmp/h200-memory-live --workers 8 --alpha 0.5
```

Run with Python importing this worktree. In this machine's shared `tl` environment, an existing editable-install hook points to the original checkout. The process-local [sitecustomize.py](results/h200-unified-memory/environment/sitecustomize.py) used for verification removes that redirect when its directory and this worktree are placed on `PYTHONPATH`; it also applies to child workers. The shared installation is unchanged.

## CPU analysis cost with diagnostics disabled

These measurements compare the original diagnostic path with version 35, before
the metadata optimization described below.

The [paired measurement](results/h200-unified-memory/lean/benchmark.json) samples
original pool indices `0`, `N//2`, and `N-1` in each case: 75 configurations,
chosen without consulting oracle winners. Each mode has one warmup and seven
timed calls on the same prebuilt PrimFunc, with alternating execution order.
The measurement includes construction of returned analysis reports and excludes
IR construction, JSON serialization, compiler lowering, and GPU execution.

Summing each candidate's median time gives **9.807 seconds with diagnostics**
and **7.587 seconds without: 22.6% less CPU analysis time, or 1.29× faster**.
The numbers below average those candidate medians within each family.

| Family | Diagnostics enabled | Lean default | Speedup |
|---|---:|---:|---:|
| Attention | 99.638 ms | 6.050 ms | 16.47× |
| GEMM | 9.036 ms | 2.035 ms | 4.44× |
| FP8 GEMM | 8.313 ms | 1.879 ms | 4.42× |
| Grouped GEMM | 503.003 ms | 491.338 ms | 1.02× |
| KDA | 33.807 ms | 4.487 ms | 7.53× |

Grouped GEMM benefited much less in version 35; eager resolution of its
input-dependent accesses remained enabled. These timings use the frozen study's
resource settings; strict register policies can still require backward
propagation. This is a local CPU analysis measurement, not an end-to-end
autotuning or GPU speedup.
The [comparison and timing script](results/h200-unified-memory/lean/verify_and_benchmark.py)
also checks lean/full score and resource-decision equality for all timed inputs.

## Deferred metadata simplification in version 36

Lean memory mode now substitutes metadata without simplifying each complete
address or predicate. It still resolves metadata lookup indices, loop bounds,
and native access extents with launch bounds, because they establish lookup
validity and scored work. All metadata loads remain counted. Read-only checks,
alias checks, and opaque-operation detection remain enabled. When deferred
collection reports uncertainty, the collector retries the existing eager path.
The optimization uses no kernel recognition or new kernel-specific helpers.

`memory_diagnostics=True` restores eager resolution together with the detailed
analyses. Timing modes also retain eager resolution. The returned
`ir_context.metadata_resolution` records `deferred`, `eager`, or `not_needed`.
The original PrimFunc remains unchanged for compilation.

The [paired benchmark](results/h200-unified-memory/metadata/benchmark.json)
compares both paths with diagnostics disabled on the same prebuilt PrimFuncs:
indices `0`, `N//2`, and `N-1` from each of the five metadata-bearing cases,
15 configurations total. Each path has one warmup and seven timed calls per
configuration, with alternating order. All 15 use deferred resolution without
fallback, and match the original scores and complete resource decisions.

Mean candidate median CPU analysis time falls from **481.393 ms to 44.149 ms**:
**10.90× faster, or 90.8% less time**. Generation, compilation, serialization,
and GPU execution are excluded; these numbers describe grouped-GEMM analysis.

| Case | Eager resolution | Deferred resolution | Speedup |
|---|---:|---:|---:|
| grouped_gemm_aligned | 274.444 ms | 41.719 ms | 6.58× |
| grouped_gemm_decode | 564.314 ms | 45.764 ms | 12.33× |
| grouped_gemm_down_aligned | 274.255 ms | 41.329 ms | 6.64× |
| grouped_gemm_prefill | 565.139 ms | 45.462 ms | 12.43× |
| grouped_gemm_ragged | 728.815 ms | 46.475 ms | 15.68× |

The [comparison and timing script](results/h200-unified-memory/metadata/verify_and_benchmark.py)
also verifies full-pool parity against version 35 and records the measured source
hashes. The eager baseline uses the original collector resolution path, with
all other code and settings shared.

## Post-compile rejection in version 37

The H200 memory autotuning runner now enables a compiler-only rejection policy.
It keeps pre-lowering settings at `mode="report_only"`,
`max_spill_bytes=None`, and `max_local_bytes=None`, and supplies separate
`post_compile_policy` limits. Thus the score, pre-lowering eligibility, and
frozen strict-alpha shortlist retain their existing behavior. After compilation,
PTXAS counters and physical limits can reject a selected configuration before
benchmarking. Rejected configurations are not replaced.

| Experiment family | Maximum spill stores | Maximum spill loads | Maximum local/stack bytes |
|---|---:|---:|---:|
| GEMM, grouped GEMM, KDA | 0 | 0 | 0 |
| Attention | 64 | 64 | 64 |
| FP8 GEMM | 128 | 128 | 128 |

These are inclusive, independent PTXAS counters, not estimates of runtime spill
traffic. The policy is explicit in
[resource_policy.py](common/resource_policy.py), using the experiment's workload
declaration outside the family-independent analyzer. It applies to the H200
branch's Hopper memory runs; other architectures and timing modes retain their
existing settings. Hardware register, block, and shared-memory limits remain
enforced. Missing observations remain unknown rather than being called zero.

Fresh compilation of all 25 frozen oracle configs found:

- Noncausal attention: 48 bytes of spill stores, 48 of spill loads, and 48 local
  bytes. The other four attention winners reported zero.
- FP8 FFN-down: 88 bytes of spill stores, 60 of spill loads, and 64 local bytes.
  FP8 prefill, square, and square-large: 92 stores, 64 loads, and 64 local bytes.
  FP8 decode reported zero.
- All 15 GEMM, grouped-GEMM, and KDA winners reported zero spills/local bytes.

Consequently, zero-spill rejection for every non-attention family would discard
four FP8 oracle winners. The declared budgets round the observed maxima upward
to 64 and 128 bytes. They are calibrated on this known study and are not a
guarantee for a different workload contract or compiler version. The runtime
does not consult oracle identities or measured latencies.

All 25 winners were compiled again through the real `TileTuneSession.post_compile`
hook with rejection enabled, and all 25 passed. Their fresh analysis scores
matched the saved 50% ranking reports. The audit did not rerun kernels for
correctness or latency: it validates compiler-resource acceptance of the
previously correctness-checked configs. See the
[committed resource summary](h200_post_compile_resources.json),
[initial resource audit](results/h200-unified-memory/post-compile/oracle-resources.json),
[enforced-filter audit](results/h200-unified-memory/post-compile/oracle-filter-validation.json),
and [compilation script](results/h200-unified-memory/post-compile/audit_oracles.py).

## Validation

- Version 37: all 25 oracle configs compile and pass the enabled compiler-resource
  policy. Generated CUDA source hashes and PTXAS counters match the report-only
  compilations, and fresh scores match the existing strict-alpha selection.
  The regression suite passed 525 tests with 38 skipped; an additional runner
  integration check verifies that H200 memory requests enable the separate
  compiler policy. [Test log](results/h200-unified-memory/post-compile/pytest.log).
- Version 36: all 29,200 fresh CPU analyses have finite scores and exactly match
  the frozen replay. The [current full-pool report](results/h200-unified-memory/metadata/live/summary.json)
  retains 25/25 oracle hits at alpha=0.5 and the worst tail rank of 90/192.
- [Exact comparison with version 35](results/h200-unified-memory/metadata/parity.json)
  confirms identical cost reports except process-local buffer identities,
  complete resource-decision reports, rankings, selections, and oracle tail
  ranks for all 29,200 configs.
- Version 35 lean validation: 29,200/29,200 fresh CPU analyses and finite scores;
  zero replay mismatches. The [version 35 full-pool report](results/h200-unified-memory/lean/live/summary.json)
  retains all 25 oracle hits at alpha=0.5.
- [Exact comparison with the version 34 reports](results/h200-unified-memory/lean/parity.json)
  confirms unchanged scores, complete rankings, selections, oracle tail ranks,
  and resource rejection decisions across all 29,200 configs. Diagnostic
  uncertainty descriptions are intentionally not treated as rejection decisions.
- All input PrimFuncs were unchanged after analysis. Family-policy constructors, family selection, pipeline timing, occupancy timing, and warp-specialization prediction were disabled by assertions.
- All 25 winners are in the strict 50% selections; every selected group fits the budget. Final saved scores, rankings, and selections were independently rechecked after formatting.
- Version 36: 500 tests passed and 38 were skipped in the TileTune suite plus the memory
  replay, common experiment, and portable-core tests. This includes lean/full
  score and rejection comparisons, strict register caps, launch limits, and
  report-only mode, plus metadata-dependent loops and access extents,
  uncertainty fallback, out-of-range metadata, and read-only metadata checks.
  [Test log](results/h200-unified-memory/metadata/pytest.log).
  Ruff and `git diff --check` passed.
- Ranking validation through version 36 used archived correctness-checked oracle
  measurements and CPU analysis while GPUs were occupied. Version 37 adds
  compiler-resource validation; no fresh GPU performance claim is made.

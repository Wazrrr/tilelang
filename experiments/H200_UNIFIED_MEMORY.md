# Unified TileTune memory ranking on H200

**All 25 oracle winners have finite scores and conservative tail ranks within the first 50% of their full pools.** Fresh CPU analysis of all **29,200 PrimFuncs** exactly matched the frozen-fact replay. The worst oracle rank is **90/192 (46.875%)**, for `grouped_gemm_decode`. Strict `alpha=0.5` selection retained exactly half of each pool, without splitting ties.

The source is the completed `h200-25shape-20260917T080241Z` FP16/E4M3 study. Kernels, inputs, configuration pools, resource policies, and exhaustive oracle measurements are unchanged. This result does not apply automatically to the newer BF16/block-scaled contract. No kernels were compiled or timed for this analysis.

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
checks are unchanged. Timing modes retain their existing analyses.

The regular runner accepts `--method top_k --metric memory --alpha 0.5`. Alpha and `top_k` are mutually exclusive in `TileTuneConfig`; the CLI alpha option overrides its default top-K budget. `strict_top_k=True` provides the same boundary policy for an explicit integer K. The original reports above use analysis version 34 and `memory.v2`. Lean analysis uses version 35 and `memory.v3`; a `null` dependency list means collection was disabled. The score formula is unchanged, and the diagnostics setting is part of the cache key.

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

Grouped GEMM benefits much less; capture and resolution of its input-dependent
accesses remain enabled. These timings use the frozen study's resource settings;
strict register policies can still require backward propagation. This is a local
CPU analysis measurement, not an end-to-end autotuning or GPU speedup.
The [comparison and timing script](results/h200-unified-memory/lean/verify_and_benchmark.py)
also checks lean/full score and resource-decision equality for all timed inputs.

## Validation

- Version 35 lean validation: 29,200/29,200 fresh CPU analyses and finite scores;
  zero replay mismatches. The [new full-pool report](results/h200-unified-memory/lean/live/summary.json)
  retains all 25 oracle hits at alpha=0.5.
- [Exact comparison with the version 34 reports](results/h200-unified-memory/lean/parity.json)
  confirms unchanged scores, complete rankings, selections, oracle tail ranks,
  and resource rejection decisions across all 29,200 configs. Diagnostic
  uncertainty descriptions are intentionally not treated as rejection decisions.
- All input PrimFuncs were unchanged after analysis. Family-policy constructors, family selection, pipeline timing, occupancy timing, and warp-specialization prediction were disabled by assertions.
- All 25 winners are in the strict 50% selections; every selected group fits the budget. Final saved scores, rankings, and selections were independently rechecked after formatting.
- 485 tests passed and 38 were skipped in the TileTune suite plus the memory
  replay, common experiment, and portable-core tests. This includes lean/full
  score and rejection comparisons, strict register caps, launch limits, and
  report-only mode. [Test log](results/h200-unified-memory/lean/pytest.log).
  Ruff and `git diff --check` passed.
- H200 GPUs were occupied, so validation used archived correctness-checked oracle measurements and CPU analysis. No fresh GPU performance claim is made.

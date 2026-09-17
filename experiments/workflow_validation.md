# Workflow validation — 2026-09-16

The system, baseline-reuse and utility-organization changes were checked on this
host using the `tl` environment and idle H200 GPUs 0 and 1. These are functional
checks using small configuration subsets, not replacement full-sweep results.

## Offline checks

The experiment tests and affected TileTune compiler-accounting tests passed:
**291 passed, 43 skipped** with GPUs hidden. Tests cover immutable baseline reuse
across TileTune runs and K values, baseline-seed invalidation, changed-artifact
rejection, measurement identities, contention rejection and discarded outcomes.

All 80 saved configuration-space audits match their pre-cleanup snapshots,
including configuration order and IDs. The two recorded full H200 GEMM heuristic
files remain unchanged. Replaying the historical GEMM Oracle@1/5/10/20/50
comparison produced the exact same JSON. Ruff and `git diff --check` passed.

## System execution

Each check used two configurations, two compilation workers, 1 ms warmup and
2 ms repetition. All accepted invocations were observed uncontended throughout.

| Final case | Original indices | Successful variants |
| --- | --- | --- |
| GEMM 4096³ | 1403, 1407 | combined |
| GEMM 8192³ | 1403, 1407 | baseline |
| Attention noncausal | 170, 172 | combined |
| Attention causal | 170, 172 | baseline |
| KDA regular | 392, 394 | baseline, pipeline, grouped, multi_gpu, combined |
| KDA unequal dimensions | 392, 394 | baseline |
| Softmax aligned | 13, 14 | baseline, pipeline, grouped, multi_gpu, combined |
| Softmax irregular | 13, 14 | baseline |

Artifacts, including requests, source hashes, numerical-check logs, timings and
GPU observations, are under:

- [First system checks](results/system-workflow-validation-1789585296088087198/)
- [System checks after the grouped-output fix](results/system-workflow-validation-1789585689328811864/)

The first grouped-softmax attempt exposed an autotuner defect: ordinary grouped
compilation ignored the eager example's `tilelang_out_idx` attribute. The shared
grouped compiler now resolves function attributes before grouping, including
output indices and pass/compile flags. The example kernel remains unchanged.
The failed attempt remains recorded. The softmax output-contract regression and
existing grouped MMA/WGMMA function-settings test both passed on H200:
[GPU regression artifacts](results/grouped-output-regression-1789585955806092748/).
The Carver adapter's three tests also passed on H200, including ranking the full
2,304-config GEMM pool for FP16 and BF16:
[Carver regression artifacts](results/carver-workflow-regression-1789586140963856018/).

## Baseline collection and reuse

A real softmax baseline bundle used configurations 13 and 14 for the two
training shapes, validation shape and both final test shapes. It performed
seeded 10% training/validation collection, trained XGBoost, saved full test
rankings, and measured both test oracle tables. Carver recorded its explicit
GEMM-only support boundary. All nine worker invocations were observed
uncontended.

Calling the collector again reused the bundle in approximately **22 ms**,
without launching experiments or changing any of its 76 JSON files. The
completion manifest verifies 75 artifacts; the additional file is the manifest.
The bundle and original costs are under
[baseline reuse validation](results/baseline-reuse-validation-1789585935244366005/).

These two-config oracle tables establish the reuse protocol, not performance
coverage of the full 224-config softmax pool. Full study runs retain the complete
declared family pools.

## Retired-family cleanup

The subsequent cleanup removed the FP8/vector experiment families, their CLI and
XGBoost adapters, vector-only helpers/presets, and the old expanded-Ampere
manifest. Shared-runner defaults now use the eight family-owned final cases
(development shapes with `--smoke`); the comparison command uses the existing
family training/validation splits by default. Compiler-accounting regression
programs remain local to `testing/python/tiletune/regression_kernels.py`.

Validation in `tl` with GPUs hidden passed **304 tests, 31 skipped**:

```bash
env CUDA_VISIBLE_DEVICES='' .agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m pytest -q testing/python/experiments \
  testing/python/tiletune/test_ampere.py \
  testing/python/tiletune/test_ownership_counting.py \
  testing/python/tiletune/test_region_schedule.py \
  testing/python/tiletune/test_targets.py
```

All 112 configuration-space audits (development, final, training and validation
cases across four native target presets) match the pre-removal snapshot exactly.
The frozen final manifest and both H200 GEMM heuristic files are unchanged.
Nine shared/family plans passed under `python -S` without importing GPU/compiler
or ML runtimes. Ruff and whitespace checks passed. This cleanup did not collect
new GPU performance measurements; the GPU checks above describe the preceding
workflow validation.

## Attention Carver adapter

The attention baseline now uses the unchanged `FlashAttentionTemplate` and
`TensorCorePolicy` through `flash_attention/carver.py`. It scores the existing
320-config pool, checks the native thread counts, and measures selections through
the ordinary example-kernel runner. No softmax/causal correction, memory-limit
increase, candidate expansion or fallback ranking is added to Carver.

Artifacts: [attention-carver-adapter-1789588721233592696](results/attention-carver-adapter-1789588721233592696/validation.json).
Both development and both final FP16 cases ran with K=20 on NVIDIA H200 using
3 warmup and 10 measurement repetitions, two compile workers, and the standard
one-second process monitor. Every invocation was observed uncontended.

| Split | Case | Selected / 320 | Result |
| --- | --- | ---: | --- |
| Development | noncausal, S=512, D=64 | 16 | All selected configs failed compilation |
| Development | causal, S=640, D=128 | 4 | All selected configs failed compilation |
| Final | noncausal, S=768, D=64 | 0 | All configs rejected by Carver's shared-memory limit |
| Final | causal, S=1152, D=128 | 0 | All configs rejected by Carver's shared-memory limit |

Development failures retain the example compiler's fragment-layout and FullRow
warp-partition diagnostics. The final minimum shared-memory estimates are
52,224 and 78,848 bytes against Carver's unchanged 49,152-byte limit. Final runs
save complete reports with `model_unavailable`, zero selected candidates and no
winner. This terminal model outcome is accepted by the reusable baseline cache;
offline Oracle@K is N/A at every K. Compilation failures remain `failed` and are
not promoted to successful reusable bundles. No attention latency/performance
claim is made from these runs.

Validation passed **307 offline tests, 33 skipped**, using the regression command
above, plus **7 tests with CUDA enabled**, including both FP16/BF16 full-pool
attention scores compared directly with the original policy and the existing
GEMM tests. Cache tests cover reuse of empty Carver selections across TileTune
revisions and N/A comparison curves. Standard-library-only planning, Ruff and
whitespace checks passed. Source hashes confirm Carver, all FlashAttention
example Python files, and attention kernel/case/pool/reference files are unchanged.

## Family-owned result storage and explicit baseline refresh

Baseline storage now defaults to `<family>/results/<GPU model>/baselines/`.
The family command's `--run-baselines` invocation collects a new bundle and
updates `current.json` only after successful completion. Ordinary TileTune runs
only load and verify that reference: missing or incompatible baselines require
an explicit rerun. Failed refreshes leave previous references and bundles intact.

Ten existing single-family result directories were relocated into GEMM,
attention and softmax folders. All migrated file contents were checked before
and after the move. The two GEMM heuristic JSONs now point to the relocated
oracle/report/monitor files; the oracle SHA256 values and measurements are
unchanged. Old path symlinks preserve absolute references in raw records.
No GPU experiment or baseline measurement was rerun for this storage change.

Validation passed **330 offline tests, 41 skipped**, using the regression command
above. Tests cover read-only reuse, missing/mismatched identities, successful
explicit refresh, failure preserving the previous baseline, and exclusion of
generated family results from source fingerprints. Eight family plans (ordinary
and explicit baseline mode for each core family) passed under `python -S`
without importing GPU/compiler/ML runtimes. Ruff and whitespace checks passed.

The [migration audit](utils/results/family-storage-migration-20260916/migration.json)
records old/new locations and verified file counts. The [result guide](RESULTS.md)
describes the current storage paths and the scope of existing measurements.

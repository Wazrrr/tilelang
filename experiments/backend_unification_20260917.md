# CUDA branch unification audit

Reviewed remote snapshots: `dev-a100` at `624b8b75`, `dev-h200` at `6574a685`,
and `dev-b200` at `2e3414d5`. H200 and B200 are checked out separately in
`/root/tilelang-dev-h200` and `/root/tilelang-dev-b200`. The integration is in
`dev-a100`; the reference branches and remote histories are unchanged.

The original uncommitted A100 work is recoverable in the Git stash named
`codex-before-backend-unification-20260917`. Its model fixes and measured
artifacts were retained. The obsolete graph adapter's clipped attention traffic
and regression coverage were moved to the current canonical Carver adapter.

## Shared contracts

| Family | Development/final cases | Ordered candidates per case | Carver template |
| --- | ---: | ---: | --- |
| GEMM | 5 / 5 | 2,304 | MatmulTemplate |
| FlashAttention | 5 / 5 | 320 | FlashAttentionTemplate |
| Native FP8 GEMM | 5 / 5 | 2,304 | FP8MatmulTemplate |
| Grouped GEMM | 5 / 5 | 192 | GroupedMatmulTemplate |

The three remote branches already shared Carver templates and the first four
families' serving shapes. A100 had only a grouped Carver adapter, without the
family registration, kernel, pool or shapes. Those definitions now match H200
and B200. The complete final matrix is 25 cases and 29,200 declared candidates
per target, before hardware support or candidate outcomes are considered.

All CUDA targets use identical mathematical workloads, dtypes, ordered pools
and config IDs. The frozen manifest matches the family registry. Final FP8
uses E4M3FN; explicit E5M2 workloads retain support. FP8 uses the authoritative
example, FP32 accumulation and FP8 output, including its embedded output index
and serialized eager elaboration. No alternative kernel was introduced.

## Historical drift corrected

| Difference | Decision |
| --- | --- |
| GEMM/FP8 development test duplicated the validation shape `(2048,4096,4096)` | Use `(1536,4096,4096)` for validation; reject overlapping mathematical splits during planning. Final test shapes are unchanged. |
| B200 FP8 removed embedded output attributes, omitted the builder lock and used broad tolerances | Retain A100's locked authoritative builder and adjacent-FP8-code plus norm checks. |
| Smoke checks assumed the newest instruction for each target | Inspect actual matrix instructions, strip comments, and require FP8 evidence for FP8 workloads. |
| Workers and runtime probes selected the first visible GPU | Select an idle device matching architecture/model inside each subprocess environment. |
| Carver result validation omitted shortlist enforcement | Apply the same fixed budget/winner membership checks as other ranked methods. |
| An empty Carver shortlist aborted baseline collection | Preserve the complete rejected ranking as `model_unavailable`, without compiling replacement candidates; verify it before completing a bundle. |
| Local attention traffic fix lived in a removed adapter | Preserve once-only Q/output traffic and clipped causal/noncausal K/V transfers in `common/carver.py`. |
| Grouped metadata was unresolved on A100 and inferred from names on B200 | Adopt H200's explicit integer-input contract; retain metadata loads, masked traffic and padded matrix work. Include metadata in analysis cache identity. |
| Masked external accesses were analyzed only on Ampere | Apply region-aware access checks on every backend. |
| Automatic TMA warp specialization was modeled only on Hopper | Include Blackwell while retaining alias, predicate and native copy-classification checks. |
| MMA operand-register accounting was restricted to Ampere | Query the selected instruction on each CUDA target; include operand fragments for MMA and exclude them for WGMMA. |
| H200/B200 KDA overwrote the input-copy destination inside the software pipeline | Retain A100's example fix, which preserves both query rounding points in the gated destination without overwriting Q. |
| A pipeline regression fixture accessed 512 reduction elements through a 128-element input | Size its input for the requested loop; preserve the original buffer sizes for shorter loops. |

Analysis version 28 invalidates old analysis-cache entries. Existing A100
instruction-specific matrix profiles, FP8 conversion rates, shared consumer
traffic, KDA recognition and ownership checks remain in place. Profile version
8 and its strict dtype checks are retained instead of adding H200's second
`gemm_rates` representation alongside `gemm_instruction_rates`.

## Differences retained or requiring separate evidence

- Device capacities, supported instructions, compiler lowering, primitive rates,
  native FP8 availability and measured heuristics legitimately differ. A100
  explicitly reports five FP8 cases unsupported. Native H200/B200 results cannot
  be established on this A100 host.
- Positive-stage Hopper KDA with a partial value tile remains unscored: its
  varying TMA traffic requires a verified per-region compiler schedule. The old
  Hopper-only path silently treated this traffic as uniform. Serial tail cases
  retain exact work checks; aligned serving cases retain supported schedules.
- B200's ideal inter-CTA L2 reuse model and uncertainty-based reranking change
  the analytical algorithm on every target; they are not hardware capability
  requirements. They were reviewed but not imported without a separate ranking
  study. The B200 reference worktree preserves them.
- B200's name-based grouped override bypasses normal resolved facts and labels
  padded traffic exact. The integration uses explicit values and region analysis
  instead. Native Blackwell timing still needs validation.
- Baseline collection CLI/storage policies differ between branches. The current
  immutable, content-addressed workflow and strict kernel/compiler/runtime
  provenance checks are retained. H200's relaxed reuse/provenance rules and
  hardcoded heuristic-export timing metadata were not imported.
- Historical result files, measured device heuristics, older case names and
  oracle indices retain their original identities. They are not relabeled as
  measurements of the new serving matrix. No tuning sweep was requested or run.

All three branches use the same native compiler sources. The KDA change above
is the only authoritative-example source difference; it is a pipeline correction,
not a reason to keep separate hardware-specific algorithms.

## Verification

- A standard-library-only audit compared all five families against both side
  worktrees. Development/final shapes and ordered candidate-pool hashes match;
  the only intentional training-contract changes are the GEMM/FP8 validation
  shapes described above. Snapshot evidence is in
  `results/backend-unification-20260917/contracts.json`.
- The broad regression run covered `testing/python/tiletune`,
  `testing/python/experiments`, and the Carver CUDA-property/recommendation tests:
  **994 passed, 16 skipped, 4 failed, 26 deselected**. The four failures identified
  the Hopper tail expectation, the out-of-bounds fixture, the GPU snapshot mock,
  and an unsupported-versus-empty-ranking expectation in the new baseline test.
  All four were corrected. A final run of the seven affected test files passed:
  **141 passed, 5 skipped, zero failures**. These counts overlap; they must not
  be added. Logs are `results/backend-unification-20260917/regression.log` and
  `results/backend-unification-20260917/affected-final.log`.
- Native A100 tests passed for supported small/boundary kernels, including
  grouped GEMM in FP16/BF16, transposed and ordinary weights, ragged groups,
  and grouped compilation's output contract. Hopper KDA and FP16/FP8 matrix
  probes cross-compiled with CUDA 12.4. Blackwell contracts and analysis were
  checked offline; native H200/B200 execution was unavailable.
- The 26 explicit exclusions were the 25 large final-shape GPU tests and the
  Blackwell probe requiring a newer CUDA compiler. No full tuning sweep or
  performance comparison was performed.
- Ruff checks passed for all changed/new Python files; `git diff --check`
  passed. Both reference worktrees remain clean. The existing A100 heuristic
  artifacts and model-fidelity report match their stashed copies byte-for-byte.

The regression environment used `/tmp/tilelang-dev-a100-env/bin/python`,
`CUDA_HOME=/root/cuda-12.4`, `CXX=/usr/bin/g++-10`, `CUDA_VISIBLE_DEVICES=0`,
and one CPU thread for OMP/MKL/OpenBLAS. The broad test selection was:

```bash
python -m pytest testing/python/tiletune testing/python/experiments \
  testing/python/carver/test_tilelang_carver_cuda_driver_properties.py \
  testing/python/carver/test_tilelang_carver_recommend_hints.py \
  -q --tb=short \
  -k 'not blackwell_mma_probe_cross_compiles and not final_example_kernels_on_gpu'
```

The final affected set was `test_ampere.py`, `test_analysis.py`,
`test_modules.py`, `test_hopper_experiments.py`, and `test_warp_specialization.py`
under `testing/python/tiletune`, plus `test_heuristic_export.py` and
`test_reusable_workflow.py` under `testing/python/experiments`.

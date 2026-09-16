# Historical organization validation: 2026-09-15

This checkpoint describes the earlier organization and kernels. The compatibility
entry points, local kernel variants and fixed-grid GEMM/attention runners have
since been removed. These measurements do not validate the current expanded
pools. See the [current experiment contracts](README.md) and preserve the linked
historical artifacts with their original source identities.

## Scope

At this checkpoint, the four primary families owned their cases, configuration spaces, kernels,
references, suite commands, and census commands. Shared execution is in
`common/`; frozen inputs are in `manifests/`. The `portable/` entry points and
the original GEMM and attention fixed-grid commands retained compatibility at that checkpoint.
Final GEMM cases are 4096³ and 8192³.

The scripts include the standalone `tiletune_core` dependency for study hashes,
attempt accounting, and native fact contracts. The broader pending compiler
model changes are separate. Ordinary studies support the committed TileTune
configuration API; optional exploration requires runtime support and reports
an explicit error when it is unavailable.

## Refactor checks

Against the saved pre-reorganization sources:

- All 420 configuration-space digests matched, including candidate ordering,
  aliases, and structural rejection audits.
- All 41 generated TIR programs matched structurally.
- All 17 numerical references matched exactly.

An isolated export of the staged files, using the committed compiler runtime,
passed 166 CPU tests. These cover family entry points, census planning,
configuration enumeration, compatibility, worker requests, selection ordering,
acceptance scope, XGBoost provenance, and the standalone core contracts.

## A100 smoke checks

Checks ran on NVIDIA A100 80GB PCIe with CUDA 12.4 and G++ 10. The full default
smoke ran in the development workspace, which also contains pending model
changes. Each family attempted the deterministic 16-candidate subset:

| Family | Correct candidates | Compile failures | Launch failures | Census |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 15/16 | 0 | 1 | 1/1 |
| FlashAttention | 11/16 | 1 | 4 | 1/1 |
| KDA | 15/16 | 0 | 1 | 1/1 |
| Softmax | 16/16 | 0 | 0 | 1/1 |

Six launches exceeded the device's shared-memory limit. One attention candidate
failed compiler layout inference. All scripts exited successfully; smoke success
requires valid candidates and instruction evidence, so it does not mean every
candidate passed. Failures consumed attempts and were retained in the reports.

Local artifacts: `results/script-smoke-20260915T041148Z/report.md` and its
per-family logs, source identities, and candidate outcomes. Generated artifacts
are ignored by Git.

After the runtime compatibility fix, all four family commands also passed a
focused GPU smoke from the isolated staged snapshot: one baseline candidate per
family, using the committed TileTune runtime. Those artifacts are in
`results/commit-snapshot-smoke-20260915/`. This verifies the script dependency
boundary; it does not replace the broader candidate-pool results above.

## Remaining validation

This work does not establish five-target performance acceptance. H200,
B200/GB200, MI355X, and Ascend 910B/A2 still require native device execution,
matching profiles, and instruction validation. Ascend also requires supplied
native family grids and an external worker. The complete development comparison
and frozen final study remain to be run.

# Five-target TileTune implementation status

## Experiment organization

The user-facing entry points now live in the kernel family folders:
[GEMM](../experiments/gemm/README.md),
[FlashAttention](../experiments/flash_attention/README.md),
[KDA](../experiments/kda/README.md), and
[softmax](../experiments/softmax/README.md).
Each owns its cases, configuration presets and constraints, implementations,
references, comparison entry point, and census entry point. The existing GEMM
and attention fixed-grid runners retain their behavior through compatibility
entry points, with their code in `tiletune/legacy.py`.

Shared execution lives in `experiments/common/`. The matrix coordinator is
`experiments/suite.py`; canonical manifests live in `experiments/manifests/`.
The old `experiments.portable` modules forward to these implementations, and
old manifest paths link to the canonical files. Family acceptance reports
identify their scope and cannot certify the complete five-target matrix.
New XGBoost contexts fingerprint the family implementation and shared sources;
archived context fingerprints retain their original interpretation.

Refactor validation preserved all 420 checked configuration-space digests,
including ordered configurations, aliases, and rejection audits. Forty-one
generated TIR programs matched structurally and 17 numerical references matched
exactly against the pre-refactor sources. A focused A100 smoke check passed
all eight candidates (one baseline and one expanded configuration per family),
including analysis, numerical correctness, and applicable instruction checks.
Artifacts are in
[`experiments/results/layout-refactor-smoke-20260915/smoke.json`](../experiments/results/layout-refactor-smoke-20260915/smoke.json).
This verifies the reorganization on A100; the five-target performance study
remains incomplete.

## Delivered

- Family-owned implementations, references, spaces, and case definitions under
  `experiments/gemm`, `flash_attention`, `kda`, and `vector`. The previous portable
  kernel and space entry points remain compatibility facades.
- The independently installable `tiletune_core` package. Numerical CUDA service,
  register-policy, residency, pipeline, compressed-region, ranking, and selection
  equations now live outside the compiler package. TIR extraction remains in
  `tilelang/tiletune`.
- Versioned `KernelFacts`, `BackendModel`, and `AnalysisReport` contracts; strict
  JSON serialization; native engine scheduling with asynchronous completion;
  explicit backend registration and missing-profile diagnostics.
- CUDA fact export through `TileTuneConfig.facts_path` and standalone replay.
  Analysis reports use version 22. Existing profile readers still accept profile
  versions 2–5. Suite requests use worker protocol version 2; archived version-1
  requests retain their original wire identities and remain readable.
- Structured pipeline diagnostic codes emitted where facts become unresolved,
  instead of categorizing failures by parsing diagnostic text.
- A compact suite runner with smoke/development/final budgets, fixed K=20,
  deterministic pairwise configuration coverage, canonical alias elimination,
  preserved original indices, and actual pool-size reporting.
- Explicit two-training/one-validation-shape XGBoost inputs per family, the
  agreed hyperparameters, three final seeds, all-seed selection before final
  oracles, shared primitive profiles/oracles, and seven shuffled winner checks.
- Per-case correctness-score coverage, per-seed acceptance, preparation/online
  costs, missing-target reporting, process leases, and resumable worker artifacts.
  Winner remeasurement can use the external worker protocol.
- A frozen eight-case FP16 holdout shape manifest at
  `experiments/portable/manifests/five_target_final.json`. It is separate from
  the existing repair manifests. Family definitions must match the manifest.

## Verification on this host

The available device is **NVIDIA A100 80GB PCIe**. Verification used CUDA 12.4,
G++ 10, and the existing Python environment (PyTorch 2.7.0+cu126). CUDA driver
575.57.08 was reported by the host. A CUDA 12.4 compiler cannot compile
`sm_100a`; cross-compilation for that architecture remains unverified here.

Thirty pre-refactor CUDA reports (Ampere/Hopper; GEMM, attention, causal
attention, KDA chunk output, softmax) retained identical scores, ranking details,
pressure, and traffic after extraction. These are analytical compatibility
checks, not Hopper device validation.

The standalone core wheel built and installed without dependencies. Evaluation
ran with Python site packages disabled and no TileLang/TVM/PyTorch imports.
Tests cover fact replay, unknowns, allocation boundaries, asynchronous waits,
compressed repetition, budget failures, deterministic subsets, frozen manifests,
worker identity, and acceptance with missing cases.

The A100 smoke run attempted four cases × 16 configurations. It performs analysis,
compilation, numerical checks, and generated-PTX inspection, without latency
comparisons. Results are retained at
`experiments/results/five-target-smoke-20260915/smoke.json` with per-candidate
outcomes, attempts, source hashes, and PTX files.

| Case | Correct candidates | Failed compilation | Failed correctness |
|---|---:|---:|---:|
| GEMM aligned | 15 | 0 | 1 |
| Attention noncausal | 11 | 1 | 4 |
| KDA chunk regular | 15 | 0 | 1 |
| Softmax aligned | 16 | 0 | 0 |

Failures consumed attempts; none were replaced or removed from the report.
These smoke artifacts are development evidence collected during implementation,
not a frozen final performance study. Each worker records its source identity.

## Remaining work — acceptance is incomplete

This change does **not** establish end-to-end five-target acceptance.

| Milestone | State |
|---|---|
| Family/suite reorganization | Implemented and exercised on A100 |
| Shared evaluator | CUDA extraction and standalone evaluation verified; native fact import contracts added |
| Native five-target execution | Incomplete |
| Development ranking gates | Not executed for the full matrix |
| Frozen final comparison | Not executed |

The native backend registrations are uncalibrated contracts. The Ascend and
CDNA4 import adapters consume already-resolved compiler records; generating
those records from native storage planning, ownership, and synchronization
passes still needs integration and verification. Blackwell's existing generic
GEMM path must be replaced by native TCGEN05/TMEM family implementations and
validated. The smoke checker explicitly requires TCGEN05 instructions on that
target, so a generic MMA fallback cannot count as native validation.

H200, B200/GB200, MI355X, and Ascend 910B/A2 device access was not available in
this session. Their calibrated profiles, complete family implementations, native
instruction validation, development studies, and final acceptance remain open.
The full eight-case A100 development comparison also remains to be run. No
Oracle@20 or online-speedup result is claimed from smoke checks.

MI355X is explicitly CDNA4/`gfx950`, as documented in the
[AMD ROCm 7.1.1 target table](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.1/reference/system-requirements.html).
The Ascend source pin is recorded in `experiments/ascend/environment.json`:
[TileLang-Ascend revision 2a06f3e](https://github.com/tile-ai/tilelang-ascend/tree/2a06f3e7ca673e6cc162c88c86d65626b5ca9b81).
The pin is a source identity, not a validated CANN environment.

## Commands

```bash
# No compiler/runtime imports or device queries:
python -m experiments.portable.suite --suite smoke --plan
python -m experiments.portable.suite --suite development --plan
python -m experiments.portable.suite --suite final --freeze --output /path/final

# A100 correctness/instruction smoke:
CUDA_HOME=/root/cuda-12.4 CXX=/usr/bin/g++-10 \
  PATH=/root/cuda-12.4/bin:$PATH .venv/bin/python -m experiments.portable.suite \
  --suite smoke --devices ampere --output /path/new-smoke

# Compact comparison, once the device is idle:
python -m experiments.portable.suite --suite development --devices ampere \
  --output /path/development

# Final execution requires passing development gates for requested targets:
python -m experiments.portable.suite --suite final \
  --device-manifest /path/devices.json --development-report /path/acceptance.json \
  --output /path/final
```

`--device-manifest` takes a JSON list of `Device` objects, including native
configuration grids, worker argv, profiles, and expected device model patterns.
An Ascend device without native family grids remains explicitly unavailable.
Do not put both compiler installations on the same Python import path. Install
`tiletune_core` into the pinned Ascend environment and launch its worker in that
environment; exchange the shared fact and worker artifacts across processes.

Final execution will not start without development gate evidence. An accepted
final aggregate requires all five targets; a one-target run cannot certify the
five-target task. Changing frozen sources, settings, shapes, or profiles requires
a new output directory. If final measurements inform model changes, freeze a new
holdout rather than reusing these final results for acceptance.

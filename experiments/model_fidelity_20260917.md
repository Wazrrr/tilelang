# Kernel model fidelity on A100

This change audits the eight current final workloads against their authoritative
example builders. Kernel algorithms, dtypes, references, pool ordering and pool
sizes are unchanged. Retired experiment families are not reintroduced.

| Family | TileTune checks | Existing Carver template reused |
| --- | --- | --- |
| FP16 GEMM | NT operands, padded tile FLOPs, reduction iterations, global traffic, shared output path | `MatmulTemplate` |
| FlashAttention | Both GEMMs, online recurrence, causal iteration bounds, once-only Q/output, shared output path | `FlashAttentionTemplate` |
| Native FP8 GEMM | FP8 inputs/output, FP32 accumulation, reduction iterations, dtype-specific output conversion | `MatmulTemplate` |

All required templates already exist in this branch. The adapters continue to
use them; no duplicate template or replacement kernel was added. GEMM rankings
now explicitly record the template name, as the other adapters already did.

## Corrections

TileTune analysis version **26** charges the shared reads/writes of scalar
consumers, internal copies and epilogues. Previously these had zero shared-byte
work. A KDA gate tile now includes FP16 Q reads, FP32 gate reads and FP16 gated-Q
writes. The FP16 kernels' fragment-to-shared and shared-to-global output copies
also consume shared service. Global input copies retain the existing producer
transfer accounting. The version change invalidates prior cached rankings.

Carver previously multiplied a complete attention-tile graph's traffic by the
key-loop count, repeating the query load and output store. Its adapter now uses
the actual BSHD transfers: Q/output once, K/V for each visited key tile, including
causal bounds and clipped query/key tails. The priority remains
`(traffic_bytes + 1) * waves`, where attention traffic is the mean across query
CTAs. Reports retain the original single-tile graph traffic separately as
`template_traffic_bytes` and expose once-only and K/V traffic.

## Verification

The new regression tests initially reproduced 11 failures. After the corrections,
all 34 focused tests passed. They include all eight final workloads with stage
counts 0 and 2, mathematical checks of matrix work and global traffic, and
full-pool Carver ranking checks for both attention and both KDA cases.

The broader A100 run passed **695 tests**, with **18 skipped** and **1 deselected**.
It covers `testing/python/tiletune`, `testing/python/experiments`, Carver template
tests and CUDA property tests. The deselected Blackwell probe requires a newer
CUDA compiler than the installed 12.4. FP8 Hopper cross-compilation and conversion
tests passed; native FP8 execution is unsupported on A100.

Using the fresh measured profile, all six FP16 final workloads completed smoke,
TileTune top-2 and Carver top-2 runs through `experiments.common.run`. Each
selection run compiled, numerically checked and benchmarked two frozen
candidates. Both FP8 cases retain explicit unsupported records for all three
methods. The final 24 case/method results are listed in
[`completed-summary.json`](results/model-fidelity-a100-20260917/completed-summary.json).

The initial smoke requests checked 14 candidates: 13 passed and one KDA
candidate could not compile. TileTune left that candidate unscored. Carver
rejected the entire initial three-config KDA subset; its complete pool still has
supported candidates. A supplemental check used existing configurations with
stages 0 and 2 for all three methods, and both candidates passed. The original
rejection remains recorded; no candidates were substituted inside a frozen run.

One supplemental invocation was rejected by the GPU monitor following a PID
ownership mismatch. Its measurements were discarded and a fresh invocation
completed. [`validation_attempts.json`](results/model-fidelity-a100-20260917/validation_attempts.json)
retains every attempt, including the rejected subset and discarded invocation.
No exhaustive oracle or cross-method performance comparison was collected.

Artifacts, including a newly measured primitive profile, requests, per-candidate
reports, generated PTX, GPU observations and test logs, are under
[`results/model-fidelity-a100-20260917/`](results/model-fidelity-a100-20260917/).
The isolated Python environment is `/tmp/tilelang-dev-a100-env`; the pinned
native sources are built in `build/` with g++ 10 and CUDA 12.4.
The broader test command was:

```bash
CUDA_HOME=/root/cuda-12.4 CXX=/usr/bin/g++-10 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/tilelang-dev-a100-env/bin/python -m pytest \
  testing/python/tiletune testing/python/experiments \
  testing/python/carver/test_tilelang_carver_experiment_templates.py \
  testing/python/carver/test_tilelang_carver_cuda_driver_properties.py \
  -q --tb=short -k 'not blackwell_mma_probe_cross_compiles'
```

That command ran through `experiments.utils.monitor.run_monitored`; its log and
observations are in `regression/`. Ruff checks and formatting pass. All four
GPUs had 0 MiB allocated and 0% utilization after validation completed.

## Follow-up instruction

Each producer buffer is released after its own last consumer. Device profile
version **8** retains separate measured MMA and WGMMA rates for Hopper, selected
per operation with unchanged dtype checks.
Profiles without a matching instruction rate still leave that operation
unscored; older profile files remain readable.

All actual GPU execution in this review used an **A100**. FP8 GEMM is unsupported
on A100 and was not executed. The FP8 family stays registered so unsupported
results remain explicit. Offline checks additionally covered Hopper code
generation and synthetic-profile model coverage; they establish neither native
FP8 correctness nor Hopper timing accuracy. Per the requested A100 scope, no
additional FP8 execution or performance study is required here.

The added tests check KDA gate consumers, producer lifetimes, instruction-specific
service rates, strict dtype matching, profile caching and malformed rate rows.
The source kernels, experiment pools and Carver templates are unchanged by this
follow-up; all needed Carver templates were already present.

The final regression invocation passed **707 tests**, with **18 skipped** and
the CUDA-12.4-incompatible Blackwell probe deselected. One frozen-source test
detected a formatting edit to `families/kda.py` during its run; after the source
settled, that test passed independently, completing **708 passing checks**.
The initial failure and the successful monitored A100 rerun are retained under
[`results/kernel-model-final-20260917T034937Z/`](results/kernel-model-final-20260917T034937Z/).
Ruff lint, formatting checks and `git diff --check` pass.

## Limits

Operation and memory checks establish agreement with the source program's
accounting; they do not establish cycle-accurate prediction. Shared scalar counts
precede compiler predication and broadcast reuse and do not model bank conflicts.
Compiler scratch registers, spills and cache effects retain their existing
uncertainty boundaries. Carver retains its own graph resource and occupancy
heuristics; attention's mean traffic does not model unequal CTA execution times
or online recurrence cycles. Unsupported configurations remain visible.

These are targeted checks, not exhaustive oracle sweeps or ranking-accuracy
measurements. Synthetic rates in unit tests exercise the model only; end-to-end
checks use the separately measured A100 primitive profile. Native FP8 numerical
and timing validation still requires compatible hardware. GPU observations are
polled once per second; this container reports host GPU PIDs that cannot always
be attributed to its local process namespace.

## Completed full A100 study

The subsequent full study completed all six supported FP16 test cases in the
current registry, with **6,688 exhaustive configurations** (5,822 valid).
Native FP8 GEMM was excluded on A100. This supersedes the smoke-only scope of
the earlier verification sections; it does not change their historical results.

Oracle@20 uses one common exhaustive latency table for each case: exhaustive-best
latency divided by the fastest successful configuration in the first 20 ranked
candidates. Failed candidates consume slots without replacement; 100% is optimal.

| Workload | TileTune | Carver | XGBoost |
| --- | ---: | ---: | ---: |
| GEMM 4096³ | 81.94% | 68.91% | 68.98% |
| GEMM 8192³ | 89.24% | 70.67% | 65.56% |
| Attention noncausal | 100.00% | No valid winner | 96.97% |
| Attention causal | 100.00% | 60.22% | 60.95% |
| KDA regular | 100.00% | 90.53% | 98.92% |
| KDA tails | 99.47% | 92.70% | 95.29% |

TileTune ran with selection seeds 123, 456 and 789; Carver and XGBoost baselines
were collected once and reused. XGBoost used separate training and validation
shapes, without final-test labels. Carver reused the existing family templates;
all 20 noncausal-attention choices failed compilation. The exhaustive winners
were remeasured seven times and exported under each family's
`heuristics/A100 80GB PCIe/` directory.

TileTune's median online tuning time was 297.90/365.75 seconds for the two GEMMs,
418.56/531.85 seconds for attention, and 89.41/93.66 seconds for KDA. Compared
with exhaustive search, the corresponding speedups were 7.23×/12.13×,
0.99×/0.97×, and 8.84×/4.54×. Primitive profiling and XGBoost preparation are
reported separately, as are discarded invocations and coordinator recovery.

KDA passes the study's acceptance gates. GEMM fails ranking-quality and model
coverage gates; attention fails coverage and tuning-speed gates. Valid-candidate
score coverage is 80.80% for GEMM, 73.22%/60.27% for attention, and 100% for KDA.
GEMM predictions do not distinguish the rasterization flag in 926 comparable
pairs, and the 4096³ oracle winner ranks 223rd. KDA's median absolute relative
latency error remains 52.81%/55.45% despite its strong top-20 selection. These
results therefore do not establish precise latency prediction for every kernel.

The [full report](results/a100-full-20260917-v27/RESULTS.md) includes online and
preparation costs, all best configurations, model diagnostics, failure counts,
and GPU monitoring limitations. The [machine-readable summary](results/a100-full-20260917-v27/summary.json)
retains per-repeat results and artifact references. An audit corrected stale
derived KDA Carver summaries from the successful saved retry, preserving the
immutable baseline bundle and the original summaries; Oracle@20 was unchanged.
No rankings or model code were changed after inspecting the final oracle.

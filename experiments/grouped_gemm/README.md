# Grouped GEMM experiments

This family follows [gemm/](../gemm/README.md) and calls
[`grouped_gemm.get_tir`](../../examples/grouped_gemm/example_grouped_gemm_fwd.py)
directly. It uses the concatenated forward example, with FP32 accumulation and
the example's masked fragment-to-global output. The adapter contains no local
TileLang kernel. Pointer-table and backward examples are outside this family.

For group sizes `batch_sizes=[M0, M1, ...]`, A has shape `(sum(Mi), K)`.
B has shape `(G, K, N)` with `transpose_b=False`, or `(G, N, K)` with
`transpose_b=True`. C concatenates the independent products along its row axis.
All named workloads use FP16; BF16 is supported too. Group sizes must be positive,
and N and K are shared by every group. The independent reference computes each
product in FP32 and casts to the input dtype. Both elementwise and relative-norm
checks use 0.01 tolerance. FLOPs are `2 * sum(Mi) * N * K`.

Use its family commands or `experiments.suite --families grouped_gemm` to run it alone.
Its frozen holdouts live in [grouped_gemm_final.json](../manifests/grouped_gemm_final.json).
Grouped GEMM is included in the default five-family/twenty-five-case matrix; historical
results retain their original scope.

## One configuration set

[`spaces.py`](spaces.py) declares 192 configurations in one deterministic
`expanded` pool:

| Parameter | Values |
| --- | --- |
| `block_M` | 64 |
| `block_N` | 32, 64, 96, 128, 192, 256 |
| `block_K` | 16, 32, 48, 64 |
| `num_stages` | 0, 1, 2, 3 |
| `threads` | 128, 256 |

The example has no autotuning grid, so this is an absolute 192-candidate
expansion. It includes the example's 64×64×64, 2-stage, 128-thread test launch
(index 60) and 64×128×64, 2-stage, 256-thread CLI launch (index 125).

`block_M` is fixed because the example receives padded group offsets as an input.
Those offsets depend on the M tile. The shared runners reuse identical tensors
across candidates, including grouped compilation and multiple GPUs. Fixing the M
tile preserves correct offsets without changing the example or its timing path.
Input generation prepares the group sizes, offsets, and padded offsets outside
kernel timing. Every declared tuning parameter reaches the example builder.

All methods use the same ordered pool. Smoke/development runs select original
indices; explicit configs must belong to this pool. There is no speculative
legality filter. Compilation and correctness failures remain recorded outcomes;
the pool size does not promise that every candidate compiles.

## Files and cases

- `cases.py`: development, final, and independent model-training workloads.
- `spaces.py`: compiler-free configuration pool and support contract.
- `kernel.py`, `reference.py`: example adapter and independent FP32 reference.
- `census.py`, `system/run.py`, `tiletune/run.py`: shared-runner entry points.
- `heuristics/<GPU>/`: audited results from completed sweeps, when available.

| Split/case | Group sizes | N | K | Transpose B |
| --- | --- | --- | --- | --- |
| Training A | 16, 32, 64 | 2048 | 7168 | False |
| Training B | 15, 31, 65 | 7168 | 2048 | True |
| Validation | 24, 40, 72 | 2048 | 7168 | False |
| Development decode | 1, 1, 2, 4 | 2048 | 7168 | False |
| Development prefill | 8, 16, 24, 32 | 2048 | 7168 | False |
| Development aligned | 32, 64, 128 | 2048 | 7168 | False |
| Development down aligned | 32, 64, 128 | 7168 | 2048 | True |
| Development ragged | 31, 47, 81, 129 | 7168 | 2048 | True |
| Final decode | 1, 2, 4, 8 | 2048 | 7168 | False |
| Final prefill | 16, 32, 48, 64 | 2048 | 7168 | False |
| Final aligned | 64, 128, 256 | 2048 | 7168 | False |
| Final down aligned | 64, 128, 256 | 7168 | 2048 | True |
| Final ragged | 63, 77, 111, 280 | 7168 | 2048 | True |

The two directions model MoE expert expansion and contraction using the
repository fused-MoE example's 7168 hidden width and 2048 expert width.

## Commands

Run from the repository root. Planning uses only the Python standard library.

```bash
python -m experiments.grouped_gemm.system.run --plan
python -m experiments.grouped_gemm.census --suite final --plan
python -m experiments.grouped_gemm.tiletune.run --suite full --device hopper --plan

# Check the two example launches through all system variants.
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m experiments.grouped_gemm.system.run --variant all \
  --config-indices 60 125 --output experiments/grouped_gemm/results/system-v1

# Collect baselines explicitly, then reuse them across TileTune revisions.
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m experiments.grouped_gemm.tiletune.run --suite full --device hopper --run-baselines
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m experiments.grouped_gemm.tiletune.run --suite full --device hopper \
  --output experiments/grouped_gemm/results/tiletune-revision-a

# Exhaustive oracle collection and audited heuristic export.
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m experiments.common.brute_force \
  --manifest experiments/manifests/grouped_gemm_final.json --device hopper \
  --output experiments/grouped_gemm/results/oracle-v1
```

System modes are `baseline`, `pipeline`, `grouped`, `multi_gpu`, and `combined`.
Here the `grouped` mode groups kernel compilations; every mode executes grouped
matrix multiplication. Multi-GPU modes require two idle devices of the same model.
The shared runners monitor contention and record per-config failures and timings.

Carver uses `GroupedMatmulTemplate`, reusing MatmulTemplate for each exact group
and summing the original traffic/wave priorities. Group metadata lookup and CTA
interleaving are outside that baseline model; ragged groups may be rejected by
its divisibility checks. Brute force,
XGBoost, and TileTune use the shared comparison workflow. `full` runs the complete
pool without claiming final acceptance; `final` requires passing development
gates. Only `--run-baselines` collects or refreshes baselines; ordinary TileTune
runs read a compatible bundle from `results/<GPU model>/baselines/`.
Reusable baseline collection runs on local CUDA. HIP and external Ascend
execution require separate device validation and, for Ascend, native schedules.
No complete sweep or tuned winner is bundled with this family.

## Validation

The 20 CPU tests in `testing/python/experiments/test_grouped_gemm.py` cover
planning without runtime imports, pool identities, frozen/disjoint workloads,
source fingerprints, seeded metadata, independent references, and FP16/BF16
structural equality with the example. The affected shared-framework regression
suite passed 167 tests. Both final workloads generated Hopper CUDA source for
indices 60 and 125. Grouped GEMM GPU correctness, grouped compilation execution,
system ablations, and baseline collection remain unverified: all local H200s
had foreign compute processes attached during validation.

```bash
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh --no-gpu -- \
  python -m pytest testing/python/experiments/test_grouped_gemm.py -k 'not on_gpu' -q

# Run on an idle GPU to include FP16/BF16 boundary and grouped-compilation checks.
.agents/skills/tl-conda-gpu-run/scripts/run_in_tl.sh -- \
  python -m pytest testing/python/experiments/test_grouped_gemm.py -q
```

TileTune receives the actual integer group sizes, offsets and padded offsets as
`input_values`, keyed by PrimFunc parameter index. The runner checks the tensors
against this contract before execution. Analysis resolves metadata in its own
IR view, counts the runtime metadata loads, full padded matrix work and masked
stores, and leaves the compiled example unchanged. See
[model contracts](../model_contracts.md).

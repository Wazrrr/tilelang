# Three-experiment plan: brute-force sweep scaling and live TileTune with alpha=0.5

## 1. Scope and common settings

The three experiments share one workload matrix and one measurement contract:

| Setting | Value |
| --- | --- |
| Manifest | `experiments/manifests/five_target_final.json` |
| Kernels | 5 (GEMM, FlashAttention, KDA intra-chunk, FP8 GEMM, Grouped GEMM) |
| Workloads | 25 final cases (5 per kernel) |
| Compiler workers | `TILELANG_AUTO_TUNING_CPU_COUNTS=64` |
| Kernel timing | CUDA events, `backend="event"` (not host wall-clock, cupti, or cudagraph) |
| Warmup / rep | `warmup=10`, `rep=50` |
| Per-config timeout | `timeout=30` (align experiments 1–3) |
| Case timeout | `case_timeout=1800` (brute force); monitored worker timeout follows the runner |
| Input seed | `seed=123` |
| Device | `ampere` (`sm_80`), 4× NVIDIA A100 80GB PCIe visible |
| Unsupported cases | FP8 GEMM is unsupported on `sm_80`; its 5 cases are recorded `unsupported`, leaving 20 executed cases |

All commands run from the repository root. Use a fresh output directory for each experiment.

## 2. Workload matrix (5 kernels × 25 workloads)

### GEMM — `bfloat16`, `transpose_b=true`, A=(M,K), B=(N,K)

| Workload | M | N | K |
| --- | ---: | ---: | ---: |
| `gemm_decode` | 256 | 4096 | 4096 |
| `gemm_prefill` | 1024 | 4096 | 4096 |
| `gemm_ffn_down` | 1024 | 4096 | 14336 |
| `gemm_square` | 4096 | 4096 | 4096 |
| `gemm_square_large` | 4096 | 14336 | 4096 |

### FlashAttention — `bfloat16`, BSHD

| Workload | batch | heads | sequence | dim | causal |
| --- | ---: | ---: | ---: | ---: | --- |
| `attention_short_causal` | 1 | 32 | 512 | 64 | true |
| `attention_batched_causal` | 2 | 16 | 2048 | 64 | true |
| `attention_noncausal` | 1 | 32 | 4096 | 128 | false |
| `attention_causal` | 1 | 32 | 4096 | 128 | true |
| `attention_long_causal` | 1 | 16 | 8192 | 128 | true |

### KDA intra-chunk — `bfloat16`, `chunk_size=64`, `sub_chunk_size=16`

| Workload | batch | heads | sequence | dim |
| --- | ---: | ---: | ---: | ---: |
| `kda_intra_short` | 1 | 32 | 2048 | 128 |
| `kda_intra_medium` | 1 | 64 | 4096 | 128 |
| `kda_intra_regular` | 1 | 32 | 8192 | 128 |
| `kda_intra_batched` | 2 | 32 | 4096 | 128 |
| `kda_intra_long` | 1 | 64 | 16384 | 128 |

### FP8 GEMM — `float8_e4m3fn`, `transpose_b=true`

| Workload | M | N | K |
| --- | ---: | ---: | ---: |
| `gemm_fp8_decode` | 256 | 4096 | 4096 |
| `gemm_fp8_prefill` | 1024 | 4096 | 4096 |
| `gemm_fp8_ffn_down` | 1024 | 4096 | 14336 |
| `gemm_fp8_square` | 4096 | 4096 | 4096 |
| `gemm_fp8_square_large` | 4096 | 14336 | 4096 |

These five are `unsupported` on A100 (`native FP8 GEMM requires CUDA sm_89 or newer`).

### Grouped GEMM — `bfloat16`

| Workload | batch_sizes | N | K | transpose_b |
| --- | --- | ---: | ---: | --- |
| `grouped_gemm_decode` | [1, 2, 4, 8] | 2048 | 7168 | false |
| `grouped_gemm_prefill` | [32]×8 | 2048 | 7168 | false |
| `grouped_gemm_aligned` | [128]×4 | 2048 | 7168 | false |
| `grouped_gemm_down_aligned` | [256]×3 | 7168 | 2048 | true |
| `grouped_gemm_ragged` | [63, 77, 111, 280] | 7168 | 2048 | true |

## 3. Kernel sources and configuration spaces

| Kernel | Experiment builder | Example source | Config keys | Pool size |
| --- | --- | --- | --- | ---: |
| GEMM | `experiments/gemm/kernel.py` | `examples/gemm/example_gemm_advanced_autotune.py` | `block_M, block_N, block_K, num_stages, thread_num, enable_rasteration` | 576 |
| FlashAttention | `experiments/flash_attention/kernel.py` | `examples/flash_attention/example_mha_fwd_bshd.py` | `block_M, block_N, num_stages, threads` | 480 |
| KDA intra-chunk | `experiments/kda/kernel.py` | `examples/kda/chunk_intra_token_parallel.py` | `block_H, num_stages, threads, block_DK` | 234 |
| Grouped GEMM | `experiments/grouped_gemm/kernel.py` | `examples/grouped_gemm/example_grouped_gemm_fwd.py` | `block_M` (fixed 64), `block_N, block_K, num_stages, threads` | 576 |

The KDA pool is the single benchmark-safe pool returned by `get_configs()`
(`num_stages=0`). `block_DK` uses `{4,8,16,32,64,128}` for power-of-two
`block_H` and `{32,64,128}` otherwise, because the slimmer slices hang for
non-power-of-two `block_H` (measured: `block_H=3` with `block_DK=4/8`), which
wedges the CUDA context. There is no separate analysis-only pool. FP8 GEMM is
unsupported on A100 and so contributes no E1/E2 pool: its five cases are
recorded `unsupported`, not swept.

Alpha budgets for `alpha=0.5` (`floor(alpha * pool_size)`):

| Kernel | Pool | Alpha budget |
| --- | ---: | ---: |
| GEMM | 576 | 288 |
| FlashAttention | 480 | 240 |
| KDA intra-chunk | 234 | 117 |
| Grouped GEMM | 576 | 288 |

## 4. Experiment 1 — brute force, multigpu=1

Exhaustive measurement of the complete declared pool, using one GPU only. This is the "no multigpu" baseline.

```bash
python -m experiments.common.brute_force \
  --device ampere \
  --gpus 0 \
  --shard-size 64 \
  --workers 64 \
  --output experiments/results/three-exp/exp1-bruteforce-mg1
```

- `--gpus 0` restricts the sweep coordinator to GPU 0, so every 64-config shard is benchmarked serially on that one GPU.
- `--workers 64` sets `TILELANG_AUTO_TUNING_CPU_COUNTS=64` inside each shard worker, so each shard compiles with up to 64 CPU workers.
- Per-config measurement uses `backend="event"`, `warmup=10`, `rep=50`, `timeout=30`.
- Grouped compilation is **off** in E1/E2 (`enable_grouped_compile=False`), as are pipelining (`use_pipeline=False`) and early stopping (`early_stop=False`); E1 uses `benchmark_multi_gpu=False`.
- The runner records `plan.json`, per-workload `config-space.json`, `outcomes.json`, shard `attempt-*/`, `gpu_observations.jsonl`, and final `summary.json` with `invocation_wall_seconds`.

## 5. Experiment 2 — brute force, multigpu=4

Same exhaustive sweep, but the coordinator shards across all four A100s.

```bash
python -m experiments.common.brute_force \
  --device ampere \
  --gpus 0 1 2 3 \
  --shard-size 64 \
  --workers 64 \
  --output experiments/results/three-exp/exp2-bruteforce-mg4
```

- `--gpus 0 1 2 3` makes the coordinator keep one shard per idle GPU, so four 64-config shards are measured concurrently.
- Everything else (pool, seeds, event timing, warmup/rep/timeout, contention rejection) is identical to experiment 1, including **grouped compilation off** (`enable_grouped_compile=False`), pipelining off, and `early_stop=False`; only `benchmark_multi_gpu` differs.
- Both experiment 1 and experiment 2 produce an independent full oracle table; either can feed the experiment 3 verification.
- Acceptance is the **union of the two oracles**: a workload counts as retained when the E1 winner *or* the E2 winner (plus that run's latency ties) lands inside the strict alpha=0.5 selection. It is not required that a single oracle satisfy every workload.

## 6. Experiment 3 — live TileTune, pipeline + multigpu=4 + grouped compile=8, alpha=0.5

This is a live TileTune run (analysis, strict alpha selection, then compile+benchmark of only the selected shortlist) with three system optimizations enabled together:

- `use_pipeline=True` — benchmark finished configs while later configs still compile.
- `benchmark_multi_gpu=True` with 4 GPUs — benchmark the shortlist across 4 devices.
- `enable_grouped_compile=True`, `group_compile_size=8` — compile 8 configs per merged device/host codegen unit.
- `TileTuneConfig(enabled=True, ranking_metric="memory", alpha=0.5, mode="reject", max_spill_bytes=0, max_local_bytes=None, input_values=case.input_values or None, report_path=...)`.

The ranking metric is the branch default `memory` (kernel-family-independent byte-waves/access-waves/pipeline-depth order). Post-compile register-spill checking is **on**. `--max-spill-bytes` is calibrated against the E1/E2 oracle winners (`experiments/results/three-exp/calibrate_spill.py`): they spill at most 48 store / 32 load bytes (only `attention_causal` and `attention_noncausal`; all other 18 spill 0), so 64 leaves the gate active without rejecting any oracle config. `--max-local-bytes` is left unset (spill check only), matching the original scope.

Planned command (after the code changes in section 7), run once per family:

```bash
for family in gemm flash_attention kda gemm_fp8 grouped_gemm; do
  python -m experiments.${family}.system.run \
    --variant combined \
    --tiletune --alpha 0.5 --metric memory \
    --group-size 8 \
    --max-spill-bytes 64 \
    --gpus 0 1 2 3 \
    --workers 64 \
    --warmup 10 --rep 50 --timeout 30 \
    --output experiments/results/three-exp/exp3-tiletune-alpha50-pipeline-mg4-grouped8/${family}
done
```

Expected behavior:

- Each family runner runs its 5 final cases sequentially; each case uses all 4 GPUs for its own shortlist benchmark.
- Per-family output layout follows the existing system runner: `<family-root>/<workload>/combined/` contains `request.json`, `experiment.json`, `compilation.json`, `benchmarks.tsv`, `tiletune.json`, `timings.tsv`, `summary.json`, and monitored `gpu_observations.jsonl`/`monitor.json`; `<family-root>/comparison.json` and `<family-root>/plan.json` summarize the family.
- `gemm_fp8` cases must be recorded `unsupported` (see required code change below) instead of failing the worker.
- `tiletune.json` contains the frozen `selection` (`selected_indices`, `requested_k=alpha_budget`) and per-config `status`, including `post_compile_rejected` for spilled configs.

## 7. Required code changes

> Status: implemented in `tilelang-dev-a100-tiletune` (see `tilelang/autotuner/tuner.py`, `tilelang/autotuner/grouped_compile.py`, `experiments/common/system.py`, and `experiments/verify_oracle_retention.py`).

All changes are in the `tilelang-dev-a100-tiletune` worktree. The h200-new worktree is the reference for the alpha/memory CLI and verification pattern.

### 7.1 `tilelang/autotuner/tuner.py` — make alpha actually trigger top-K preparation

Current code gates selection preparation on `self.tiletune_args.top_k is not None`. With `alpha=0.5` and `top_k=None`, the full pool would be compiled instead of the alpha shortlist. Port the h200-new fix:

```python
# before
if self.tiletune_session is not None and self.tiletune_args.top_k is not None:
# after
if self.tiletune_session is not None and self.tiletune_session.requested_k is not None:
```

`TileTuneSession.requested_k` already resolves `alpha_budget(...)` when `alpha` is set, so no other change is needed in this file.

### 7.2 `tilelang/autotuner/grouped_compile.py` — per-config fallback when a group fails at merged codegen

The grouped compiler already isolates elaboration/lowering/filter failures per config. The remaining gap is the merged stage: if `device_codegen`, `host_codegen`, module import, or JIT initialization throws, the final `except` currently marks **all** still-unfinished configs in that group with the same error.

Fix the final `except Exception as e` so that when `len(remaining) > 1` it retries each remaining config as a singleton group, and only records the group error directly for a single remaining config:

```python
except Exception as e:
    completed = {result[0] for result in unit_results}
    remaining = [item for item in lowered_items if item["idx"] not in completed]
    if len(remaining) <= 1:
        for item in remaining:
            unit_results.append((item["idx"], item["config_arg"], None, e))
    else:
        for item in remaining:
            try:
                unit_results.extend(
                    compile_grouped_unit_tvm_ffi(
                        [(item["idx"], item["config_arg"])],
                        compile_args,
                        elaborate_func,
                        filter_config=filter_config,
                        tiletune_session=tiletune_session,
                    )
                )
            except Exception as singleton_error:
                unit_results.append((item["idx"], item["config_arg"], None, singleton_error))
```

This ensures one bad config in a group of 8 cannot fail the other 7. Singletons must not recurse into the same fallback.

### 7.3 `experiments/common/system.py` — TileTune + alpha/memory + unsupported handling

Extend the system runner with:

- New CLI flags:
  - `--tiletune` (store_true)
  - `--alpha` (float, default None)
  - `--metric` (choices `memory`, `traffic_waves`, `pipeline_time`, default `memory`)
- Add `tiletune`, `alpha`, and `metric` to the frozen `settings` written to `plan.json` and `request.json`.
- In `worker`, immediately after the workload is parsed and before `case = make_case(w)` (so unsupported families never reach `make_case`):

```python
from experiments.common.spec import Device, support_reason
from tilelang.tiletune import TileTuneConfig

target = current_target()
compiler_target = target.compiler_target() if hasattr(target, "compiler_target") else {"kind": "cuda", "arch": "sm_80"}
reason = support_reason(w, Device(name="ampere", target=compiler_target))
if reason:
    write_json(output / "summary.json", dict(status="unsupported", workload=w.name, variant=request["variant"], reason=reason))
    return
```

- When `request["settings"].get("tiletune")` is true, add to the tuner chain:

```python
tiletune_config = TileTuneConfig(
    enabled=True,
    ranking_metric=settings.get("metric", "memory"),
    alpha=settings["alpha"],
    mode="reject",
    max_spill_bytes=0,      # post-compile register-spill check ON
    max_local_bytes=None,   # keep this run scoped to the spill check
    input_values=case.input_values or None,
    report_path=str(output / "tiletune.json"),
)
tuner = tuner.set_tiletune_args(tiletune_config)
```

- Keep `tuner.run(..., early_stop=False, use_pipeline=pipeline, enable_grouped_compile=grouped, group_compile_size=settings["group_size"], benchmark_multi_gpu=multi_gpu, benchmark_devices=devices)`.

This also requires `alpha` validation in `main`: finite, `0 < alpha <= 1`, and `--tiletune` required when `--alpha` is supplied.

### 7.4 New verification script `experiments/verify_oracle_retention.py`

After experiment 3 finishes, verify that every fresh oracle-winning config is retained in the frozen TileTune alpha selection.

Inputs:

```bash
python -m experiments.verify_oracle_retention \
  --oracle-root experiments/results/three-exp/exp2-bruteforce-mg4 \
  --tiletune-root experiments/results/three-exp/exp3-tiletune-alpha50-pipeline-mg4-grouped8 \
  --alpha 0.5 \
  --output experiments/results/three-exp/oracle-retention.json
```

Logic per workload:

1. Load the brute-force oracle records from `<oracle-root>/<workload>/outcomes.json` via `experiments.utils.results.load_oracle`.
2. Collect every config whose measured `latency_ms` equals the oracle minimum (winner plus any latency ties).
3. Load `<tiletune-root>/<family>/<workload>/combined/tiletune.json` (glob `**/tiletune.json` under each family root to stay robust) and map each oracle-optimal config through `config_key` to its record.
4. A config is retained iff its record has `selected == true` and its ranking entry has `tie_last_rank <= alpha_budget` (strict alpha keeps complete score groups inside the budget).
5. `hits = retained_count / oracle_optimal_count`; exit non-zero unless every supported workload has all oracle-optimal configs retained and every unsupported workload is `unsupported`. Under the E1/E2 union rule, run the check once per oracle and accept a workload when either report retains it (see `experiments/results/a100-tiletune/verify_tail_rank.py --combine`).

This mirrors `experiments/results/a100-tiletune/run_memory.py` / h200-new `experiments/results/a100-unified-memory/run_memory_a100.py`, but joins against the fresh brute-force oracle from experiments 1/2 instead of the historical bundle.

## 8. E2E timing rules

Record and report these wall-clock scopes separately:

| Method | E2E scope |
| --- | --- |
| Brute force (exp 1/2) | `summary.json.invocation_wall_seconds` (coordinator) plus per-shard `worker_seconds`; this is the full oracle collection time |
| TileTune (exp 3) | per-workload `summary.json.tuning_seconds` (analysis + compile + benchmark, including pipeline overlap) summed over supported workloads; no primitive-profile preparation is charged for `memory` metric |
| XGBoost (when compared) | **training must be counted**: `training_collection_seconds + validation_collection_seconds + fit_seconds` from the model artifact `training` block, plus the online `tuning_seconds` (model load + feature extraction + prediction + compile + benchmark) |

Do not report a pretrained XGBoost model as zero-cost. The existing `experiments/common/acceptance.py` already records XGBoost training costs as `preparation_seconds` and `first_use_seconds`; reuse that convention in the final report.

## 9. Acceptance criteria

1. Experiments 1 and 2 both produce complete terminal outcomes for all 20 supported A100 cases and explicit `unsupported` records for the 5 FP8 cases.
2. Experiment 3 finishes with `tiletune.json` per supported case, a frozen alpha selection of `requested_k == floor(0.5 * pool_size)` (complete score groups only), and no `Auto-tuning failed` summary.
3. Grouped compilation never drops an entire group of 8 because of a single failing config (verified in `compilation.json` and worker logs).
4. Post-compile register-spill check is active: spilled selected configs are recorded `post_compile_rejected`, not silently benchmarked.
5. `experiments.verify_oracle_retention` exits 0: **all** fresh oracle-optimal configs (winners and latency ties) are inside the strict alpha=0.5 TileTune selection for all 20 supported cases, where a case passes when **either** the E1 or the E2 oracle-optimal set is retained (union rule).
6. E2E report lists brute-force wall time, TileTune wall time, and (if included) XGBoost e2e with training time counted.

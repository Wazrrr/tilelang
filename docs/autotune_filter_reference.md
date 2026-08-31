# Autotune Filter Reference

This document describes the rule-based autotune filter system in `tilelang/autotuner/filters`.

The current CUDA filter is split into:

- **Common filters**: apply broadly to compiled kernels, except where a family explicitly opts out.
- **Primitive filters**: target lower-level CUDA/TileLang primitive choices such as WGMMA, TMA, K-loop shape, and tile area.
- **Kernel-family filters**: target patterns that only make sense for a kernel family, such as attention state size or quantized GEMM dequantization footprint.

The implementation entry point is `tilelang/autotuner/filters/verify.py`. Kernel-family classification and tag expansion are in `tilelang/autotuner/filters/classifier.py`.

Built-in rule definitions are split by layer:

- `tilelang/autotuner/filters/rule_sets/common.py`: common post-compile resource filters.
- `tilelang/autotuner/filters/rule_sets/primitive.py`: primitive WGMMA, TMA, K-loop, and tile-shape filters.
- `tilelang/autotuner/filters/rule_sets/kernel_family.py`: GEMM, quantized GEMM, sparse GEMM, and attention-family filters.
- `tilelang/autotuner/filters/rule_sets/__init__.py`: default rule ordering.

## Runtime Flow

1. Autotune lowers each candidate to device IR.
2. If `filter.enabled` is true, TileLang emits CUDA source with `compile_device=False`. This is still before the real NVCC/PTXAS compile.
3. `evaluate_pre_compile_filter` extracts source/IR features and runs pre-compile rules.
4. If the candidate survives, TileLang performs the real CUDA compile.
5. If any post-compile rule needs PTXAS usage, TileLang captures PTXAS verbose output and parses registers, spills, and stack frame size.
6. `evaluate_post_compile_filter` runs post-compile rules.
7. The candidate is benchmarked only if there is no hard violation and either there are no advisory findings or `filter.action="report"` is used.

`filter.action` only controls advisory rules. Hard rules always reject when they fire.

## Violation vs Advisory Rules

Each rule has one `finding_kind`:

- `violation`: a hard filter. It writes to `details["violations"]` and rejects the candidate.
- `advisory`: an advice filter. It writes to `details["advisories"]`.

For advisory rules:

- `filter.action="reject"` applies the advice and rejects matching candidates.
- `filter.action="report"` records the advice but keeps matching candidates.

This removes duplicated hard/advisory rule pairs. For example, there is one `primitive.wgmma_n` rule. It is advisory by property, so the same threshold can be used either as a pruning rule or as a report-only warning.

## Classification

The classifier returns:

- `primary_kernel_type`: the main family, for example `dense_gemm`, `flash_attention`, or `conv`.
- `kernel_type_tags`: broader tags used by rules, for example `dense_gemm -> ("matmul", "gemm", "dense_gemm")`.
- `traits`: detected primitive/source traits such as `uses_wgmma`, `uses_tma`, `has_attention_state`, `has_dequant`, or `uses_im2col`.
- `evidence`: strings explaining why the classifier chose those traits.

Explicit user hints are reserved:

- `kernel_type != "auto"` overrides automatic family detection.
- `kernel_traits` are merged into detected traits.

Current tag expansion lives in `classifier.py`:

| Primary type | Tags |
|---|---|
| `dense_gemm` | `matmul`, `gemm`, `dense_gemm` |
| `quantized_gemm` | `matmul`, `gemm`, `quantized_gemm` |
| `blockscaled_gemm` | `matmul`, `gemm`, `quantized_gemm`, `blockscaled_gemm` |
| `sparse_gemm` | `matmul`, `gemm`, `sparse_gemm` |
| `gemv` | `matmul`, `gemv` |
| `attention` | `attention` |
| `flash_attention` | `attention`, `flash_attention` |
| `flash_decoding` | `attention`, `flash_decoding` |
| `attention_sink` | `attention`, `attention_sink` |
| `block_sparse_attention` | `attention`, `block_sparse_attention` |
| `mla` | `attention`, `mla` |
| `nsa` | `attention`, `nsa` |
| `conv` | `conv` |
| `topk` | `reduction`, `topk` |
| `scan` | `scan` |
| `linear_attention` | `linear_attention` |

## Threshold Policy

Current thresholds are policy defaults, not derived automatically by the framework.

Recommended workflow for changing a threshold:

1. Run the target family with `filter.action="report"`.
2. Collect `filter.tsv` and benchmark results.
3. Join each candidate's metrics with latency.
4. Check whether advisory candidates ever include top-performing configs.
5. Keep uncertain performance heuristics as advisory rules.
6. Use hard rules only for signals that should always stop a candidate, such as register spills or missing required sparse-mask usage.
7. Recalibrate per architecture and family when needed.

`None` means no numeric limit is enforced for that rule, even if the check flag is true.

## Common Filters

Common filters use PTXAS output, so they run at `post_compile`.

| Rule name | Kind | Where | Applies to | Detects | Config fields | Default threshold | Meaning | How to decide threshold |
|---|---|---|---|---|---|---|---|---|
| `common.spills` | Hard | `rule_sets/common.py` | Non-attention kernels | `info.n_spills`, parsed from PTXAS `spill stores bytes / 4` | `check_spills`, `max_spills` | `max_spills=0` | Rejects kernels that spill registers to local memory. For GEMM-like kernels, any spill usually means the candidate is not worth benchmarking. | Start strict at `0` for dense GEMM and similar compute kernels. Relax only if data shows spilled candidates are competitive. |
| `common.local_memory` | Hard | `rule_sets/common.py` | Non-attention kernels | `info.local_size_bytes`, parsed from PTXAS stack frame size | `check_local_memory`, `max_local_size_bytes` | `max_local_size_bytes=0` | Rejects kernels that allocate stack/local memory. This often indicates register pressure or non-scalarized local arrays. | Start strict at `0` for GEMM. Relax for families where small stack frames are common and not latency-correlated. |
| `common.registers` | Hard | `rule_sets/common.py` | All kernels | `info.n_regs`, parsed from PTXAS used-register count | `check_registers`, `max_registers_per_thread` | `None` | Optional cap on registers per thread. High register use can reduce occupancy, but it is not always bad. | Do not set globally. Pick family/device-specific caps from occupancy and latency data. |

## Primitive Filters

Primitive filters use IR, CUDA source, and config-derived metrics, so they run at `pre_compile`.

| Rule name | Kind | Where | Applies to | Detects | Config fields | Default threshold | Meaning | How to decide threshold |
|---|---|---|---|---|---|---|---|---|
| `primitive.output_elements_per_thread` | Advisory | `rule_sets/primitive.py` | Non-attention kernels | `output_elements_per_thread = block_M * block_N / threads` when exactly divisible | `check_output_elements_per_thread`, `max_output_elements_per_thread` | `256` | Flags candidates that assign too much output tile work to each thread. This is a rough pressure signal for accumulators and stores. | Set by family and dtype. Use report mode to find when large per-thread output work correlates with slow or failed candidates. |
| `primitive.wgmma_n` | Advisory | `rule_sets/primitive.py` | All kernels using WGMMA | `max_wgmma_n`, extracted from IR WGMMA op text and CUDA/PTX WGMMA shapes | `check_wgmma_n`, `max_wgmma_n` | `128` | Flags large WGMMA-N choices. These can be inefficient or register-heavy, but are not universally invalid. | Default is a warning/pruning heuristic. Tune per architecture and family. |
| `primitive.k_loop` | Advisory | `rule_sets/primitive.py` | All kernels with detected K-loop or config K/block_K | `max_k_loop_iterations`, from CUDA `for (int k = 0; k < N; ...)` or `ceildiv(K, block_K)` | `check_k_loop`, `max_k_loop_iterations` | `64` | Flags very long K loops. Too many iterations can imply poor `block_K` choice and loop/synchronization overhead. | Pick from family sweeps over `K` and `block_K`. |
| `primitive.tma_store_count` | Advisory | `rule_sets/primitive.py` | Non-attention kernels | `tma_store_count`, counted from CUDA source `tl::tma_store(` | `check_tma_store_count`, `max_tma_store_count` | `None` | Optional cap on the number of TMA stores in a kernel. | No global default. Only set for kernels where too many TMA stores clearly create overhead or scheduling pressure. |
| `primitive.tma_tiny_tile` | Advisory | `rule_sets/primitive.py`, `TmaTinyTileRule` | Kernels using TMA load | `tile_area = block_M * block_N`, `tma_load_count`, and `num_stages` | `check_tma_tiny_tile`, `max_tma_tiny_tile_area`, `tma_tiny_tile_num_stages` | `tile_area <= 4096` and `num_stages == 1` | Flags tiny TMA tiles with shallow pipelines. TMA setup cost is often not worthwhile for very small tiles. | Start from the default as a conservative heuristic. Recalibrate if small-tile TMA is useful on a backend or family. |

## Kernel-Family Filters

### Dense GEMM And GEMM-Like Kernels

| Rule name | Kind | Where | Applies to | Detects | Config fields | Default threshold | Meaning | How to decide threshold |
|---|---|---|---|---|---|---|---|---|
| `gemm.c_local` | Advisory | `rule_sets/kernel_family.py` | Non-attention kernels, including GEMM-like kernels | `c_local_floats`, the largest CUDA local array matching `C_local` or `Ct_local` | `check_c_local`, `max_c_local_floats` | `256` | Flags candidates with large accumulator fragments per thread. This is a strong register-pressure/spill-risk signal. | Start at `256` for dense GEMM. For new GEMM variants, run report mode and verify that larger accumulator fragments do not produce top latency. |

Dense GEMM currently relies on:

- `gemm.c_local`
- common post-compile spill/local-memory/register rules
- primitive WGMMA/TMA/K-loop/tile-area rules

There is no separate `dense_gemm.*` rule yet beyond the generic GEMM accumulator rule.

### Quantized And Blockscaled GEMM

| Rule name | Kind | Where | Applies to | Detects | Config fields | Default threshold | Meaning | How to decide threshold |
|---|---|---|---|---|---|---|---|---|
| `quantized_gemm.dequant_elements` | Advisory | `rule_sets/kernel_family.py` | `quantized_gemm` and `blockscaled_gemm` tags | `quant_dequant_elements_per_thread`, largest local array whose name includes `dequant` | `check_quant_dequant_elements_per_thread`, `max_quant_dequant_elements_per_thread` | `128` | Flags heavy per-thread dequantization fragment footprint. | Use report mode to study dequant pressure before relying on it as a pruning rule. |

Quantized and blockscaled GEMM also use the dense/common/primitive rules when their tags include `gemm`.

### Sparse GEMM

| Rule name | Kind | Where | Applies to | Detects | Config fields | Default threshold | Meaning | How to decide threshold |
|---|---|---|---|---|---|---|---|---|
| `sparse_gemm.sparse_mask_required` | Hard | `rule_sets/kernel_family.py`, `SparseMaskRequiredRule` | `sparse_gemm` tag | Missing CUDA `BlockMask`, `block_mask`, or `blockMask` access | `check_sparse_mask` | Disabled by default | Optional correctness/sanity check for sparse GEMM candidates expected to guard work by mask. | Enable for sparse GEMM experiments where every valid candidate must use a block mask. |

Sparse GEMM also uses common and primitive filters.

### Attention Families

Attention tags include `attention`, `flash_attention`, `flash_decoding`, `attention_sink`, `block_sparse_attention`, `mla`, and `nsa`.

| Rule name | Kind | Where | Applies to | Detects | Config fields | Default threshold | Meaning | How to decide threshold |
|---|---|---|---|---|---|---|---|---|
| `attention.state_elements` | Advisory | `rule_sets/kernel_family.py` | All kernels tagged `attention` | `attention_state_elements_per_thread = acc_s + acc_o + softmax_state`, extracted from CUDA local arrays | `check_attention_state_elements_per_thread`, `max_attention_state_elements_per_thread` | `256` | Flags attention candidates with too much per-thread score/output/softmax state. This catches register pressure before NVCC. | Calibrate per attention variant and head dimension. Larger `block_M`, `block_N`, or head dim may require a different cap. |
| `attention.spills` | Hard | `rule_sets/kernel_family.py` | All kernels tagged `attention` | `n_spills`, parsed from PTXAS spill stores | `check_attention_spills`, `max_attention_spills` | `128` | Attention gets a relaxed hard spill threshold because some useful attention kernels may spill a little. | Lower it if any spill is always bad; raise it only if top configs spill more than 128 dwords. |
| `attention.local_memory` | Hard | `rule_sets/kernel_family.py` | All kernels tagged `attention` | `local_size_bytes`, parsed from PTXAS stack frame size | `check_attention_local_memory`, `max_attention_local_size_bytes` | `256` | Allows a small local stack frame for attention but rejects larger stack usage. | Calibrate by attention family. Some generated attention code may tolerate small stack frames. |

There are currently no separate hard rules for `flash_decoding`, `attention_sink`, `mla`, or `nsa`. They all inherit the attention rules through the `attention` tag. This is intentional for now: classification is in place, but variant-specific thresholds should be added only after experiments show distinct failure modes.

### Conv

Conv is classified by `uses_im2col` / `im2col` source signals and gets the `conv` tag.

Current behavior:

- Uses common post-compile filters.
- Uses primitive filters such as `primitive.output_elements_per_thread`, `primitive.wgmma_n`, `primitive.k_loop`, and TMA rules when their metrics exist.
- Does not yet have a `conv.*` kernel-family rule.

Potential future conv-specific rules:

- im2col expansion footprint
- filter/window tile reuse
- channel tile shape constraints

### GEMV

GEMV is classified from signals such as `gemv`, `alloc_reducer`, `finalize_reducer`, `o_reducer`, `C_accum`, or `reduce_threads`.

Current behavior:

- Uses common post-compile filters.
- Does not yet have a `gemv.*` family-specific rule.

Potential future GEMV-specific rules:

- reducer replication footprint
- `reduce_threads` range
- vector load width / coalescing constraints

### TopK

TopK is classified from `topk`, `top_k`, `topk_indices`, `topk_gates`, or `max_idx` signals.

Current behavior:

- Uses common post-compile filters.
- Gets `reduction` and `topk` tags.
- Does not yet have a `topk.*` family-specific rule.

Potential future TopK-specific rules:

- top-k value versus local heap/register footprint
- reduction tree size
- index/value local array footprint

### Linear Attention / KDA / GDN

Linear attention is classified from strong sequence/state signals such as `linear_attention`, `mamba`, `chunk`, `kda`, `gdn`, `gla`, `wy_fast`, `delta_h`, `prev_state`, `da_cumsum`, and `cumsum`.

Current behavior:

- Uses common post-compile filters.
- Does not inherit the `attention` tag, because linear attention kernels often have different structure from softmax attention.
- Does not yet have a `linear_attention.*` family-specific rule.

Potential future linear-attention rules:

- chunk size constraints
- recurrent state footprint
- scan/reduction local memory footprint
- forward/backward-specific state checks

## Metric Extraction Details

| Metric | Source |
|---|---|
| `n_regs` | PTXAS `Used N registers` |
| `n_spills` | PTXAS `spill stores bytes / 4` |
| `local_size_bytes` | PTXAS `stack frame` bytes |
| `c_local_floats` | CUDA local arrays named like `C_local` or `Ct_local` |
| `quant_dequant_elements_per_thread` | Largest local array whose name includes `dequant` |
| `sparse_mask_access_count` | Counts `BlockMask`, `block_mask`, and `blockMask` in CUDA source |
| `attention_score_elements_per_thread` | CUDA local array `acc_s` |
| `attention_output_elements_per_thread` | CUDA local array `acc_o` |
| `attention_softmax_elements_per_thread` | CUDA local arrays like `logsum`, `scores_max`, `scores_sum`, and related online-softmax state |
| `attention_state_elements_per_thread` | `acc_s + acc_o + attention_softmax_state` |
| `max_wgmma_n` | Max N dimension from WGMMA shape in device IR or CUDA/PTX source |
| `max_k_loop_iterations` | CUDA K-loop bound or `ceildiv(K, block_K)` from config |
| `tma_load_count` | Count of `tl::tma_load(` in CUDA source |
| `tma_store_count` | Count of `tl::tma_store(` in CUDA source |
| `output_elements_per_thread` | `block_M * block_N / thread_num` or `/ threads` |
| `tile_area` | `block_M * block_N` |
| `num_stages` | Config `num_stages` |

## Current Family-Specific Coverage

| Family | Has own family-specific filters? | Current targeted filters |
|---|---:|---|
| Dense GEMM | Partial | `gemm.c_local`, plus primitive/common |
| Quantized GEMM | Yes | `quantized_gemm.dequant_elements`, plus GEMM/common/primitive |
| Blockscaled GEMM | Yes | Same quantized GEMM rules via `quantized_gemm` tag |
| Sparse GEMM | Yes | `sparse_gemm.sparse_mask_required`, plus GEMM/common/primitive |
| Flash attention | Yes | attention state/spill/local-memory rules, plus primitive/common |
| Flash decoding | Shared attention rules | No decoding-specific thresholds yet |
| Attention sink | Shared attention rules | No sink-specific thresholds yet |
| Block-sparse attention | Shared attention rules | No block-sparse-attention-specific thresholds yet |
| MLA | Shared attention rules | No MLA-specific thresholds yet |
| NSA | Shared attention rules | No NSA-specific thresholds yet |
| Conv | No | common/primitive only |
| GEMV | No | common only |
| TopK | No | common only, plus `reduction`/`topk` classification |
| Linear attention/KDA/GDN | No | common only |

The framework now has the tags needed to add family-specific filters without changing hardcoded dispatch. The missing pieces are empirical thresholds and family-specific metrics for conv, GEMV, TopK, and linear attention.

## Deprecated Names

`tilelang/autotuner/filters/quality.py` only provides compatibility aliases for old names such as `evaluate_tir_quality_filter` and `set_quality_filter_args` style usage. New code should use `filter`, `pre_compile`, and `post_compile` naming.

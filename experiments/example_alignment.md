# Experiment source alignment

Each experiment calls its example builder directly; scheduling choices do not
select a replacement mathematical operation.

| Family | Authoritative example |
| --- | --- |
| GEMM | `examples/gemm/example_gemm_advanced_autotune.py` |
| FP8 GEMM | `examples/gemm_fp8/example_tilelang_gemm_fp8.py`; original grid in `example_gemm_fp8_tiletune.py` |
| Grouped GEMM | `examples/grouped_gemm/example_grouped_gemm_fwd.py` |
| FlashAttention | `examples/flash_attention/example_mha_fwd_bshd.py` |
| KDA intra-chunk | `examples/kda/chunk_intra_token_parallel.py` |

The original GEMM and FP8 grids each contain 288 configs. KDA has 32 original
configs, attention has one, and the grouped example supplies a fixed default.
Tests check inclusion and PrimFunc structural identity. Compilation qualification
is separate from numerical correctness and oracle-retention validation.

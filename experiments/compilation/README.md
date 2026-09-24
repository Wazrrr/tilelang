# H200 compilation qualification

Each JSON certificate identifies the same candidate grid as its family's
`spaces.candidate_configs()`. `spaces.get_configs()` selects only the recorded
successful intersection. New candidate grids or changed kernel sources require
qualification again; they are never silently admitted to the pool.

The validator builds the actual experiment PrimFunc and runs TileLang lowering
and CUDA device compilation (NVCC/PTXAS) for every retained config on all five
final shapes. Configurations that fail are removed from later shapes, since
they cannot belong to the intersection. Their errors remain in the evidence
directory. Every published pool must exceed 500 and retain the original example
grid (or the grouped example's explicit default).

Certificates record source fingerprints, native-library hashes, NVCC version,
config indices, workload definitions, and hashes of the successful output
records. Detailed per-config outcomes and diagnostics remain at each
`evidence_directory`. The original 32-config KDA grid and both 288-config GEMM
grids are checked before publication.

## Current qualification

All retained configurations compiled on all five final workloads in their
family with target SM90a and NVCC 12.9.86, with GPUs hidden.

| Family | Retained pool | Original configs included | Successful config/workload compilations |
| --- | ---: | ---: | ---: |
| GEMM | 576 | 288 | 2,880 |
| FP8 GEMM | 576 | 288 | 2,880 |
| Grouped GEMM | 576 | 1 explicit default | 2,880 |
| FlashAttention | 512 | 1 | 2,560 |
| KDA intra-chunk | 645 | 32 | 3,225 |
| Total | | | 14,425 |

## Reproduce without GPU execution

Use the repository's `tl` environment and keep CUDA devices hidden. For example:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 TVM_NUM_THREADS=1 python -m experiments.validate_compilation \
  --family gemm_fp8 --workers 128 --publish \
  --output /path/to/new-compilation-evidence/gemm_fp8
```

The command handles one family and one workload at a time, chooses CPU affinity,
leases the host against another experiment launcher, and writes each outcome
atomically. Repeating the command resumes completed configs only when the source,
compiler, candidate grid, and workload identity still match. Run families
sequentially. Publication occurs only after all five workloads finish.

These certificates prove device compilation for the recorded final shapes and
compiler. They do not prove GPU launch, numerical correctness, performance,
post-compile-filter survival, or oracle retention. GPU measurements remain a
separate stage. Other dtypes, shapes, compiler revisions, and backends require
separate qualification.

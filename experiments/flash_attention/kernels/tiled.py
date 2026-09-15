"""Additional scheduling choices owned by this kernel family."""

import tilelang.language as T
from experiments._kernel import positive_integer, gemm_policy, _row_layout


def attention_program(batch, heads, sequence, dim, causal, dtype, block_m, block_n, stages, threads, qk_policy, pv_policy, copy_width):
    qk, pv = gemm_policy(qk_policy), gemm_policy(pv_policy)
    for name, value in dict(block_M=block_m, block_N=block_n, threads=threads).items():
        positive_integer(name, value)
    if type(stages) is not int or stages < 0:
        raise ValueError("num_stages must be a nonnegative integer")
    if copy_width is not None and (type(copy_width) is not int or copy_width not in (1, 2, 4, 8)):
        raise ValueError("copy_width must be None, 1, 2, 4, or 8")
    shape = (batch, sequence, heads, dim)
    scale = dim**-0.5 * 1.44269504
    softmax_vector = min(copy_width or 1, max(1, block_m * block_n // threads))
    row_threads = threads // min(threads, block_n // softmax_vector)
    score_layout, reduction_layout = _row_layout(block_m, block_n, threads, softmax_vector, row_threads)

    @T.prim_func
    def kernel(Q: T.Tensor(shape, dtype), K: T.Tensor(shape, dtype), V: T.Tensor(shape, dtype), O: T.Tensor(shape, dtype)):
        with T.Kernel(T.ceildiv(sequence, block_m), heads, batch, threads=threads) as (bx, by, bz):
            q = T.alloc_shared((block_m, dim), dtype)
            k = T.alloc_shared((block_n, dim), dtype)
            v = T.alloc_shared((block_n, dim), dtype)
            probabilities = T.alloc_shared((block_m, block_n), dtype)
            scores_shared = T.alloc_shared((block_m, block_n), "float32")
            rescale_shared = T.alloc_shared((block_m,), "float32")
            total_shared = T.alloc_shared((block_m,), "float32")
            out_shared = T.alloc_shared((block_m, dim), dtype)
            scores = T.alloc_fragment((block_m, block_n), "float32")
            scores_mma = T.alloc_fragment((block_m, block_n), "float32")
            out = T.alloc_fragment((block_m, dim), "float32")
            maximum = T.alloc_fragment((block_m,), "float32")
            previous = T.alloc_fragment((block_m,), "float32")
            rescale = T.alloc_fragment((block_m,), "float32")
            partial = T.alloc_fragment((block_m,), "float32")
            total = T.alloc_fragment((block_m,), "float32")
            T.annotate_layout(
                {
                    scores: score_layout,
                    maximum: reduction_layout,
                    previous: reduction_layout,
                    rescale: reduction_layout,
                    partial: reduction_layout,
                    total: reduction_layout,
                }
            )
            T.copy(Q[bz, bx * block_m : (bx + 1) * block_m, by, :], q, coalesced_width=copy_width)
            T.clear(out)
            T.clear(total)
            T.fill(maximum, -T.infinity("float32"))
            # Pipelined schedules need a fixed trip count to keep prefetch/drain
            # indices aligned for short causal prefixes. Future tiles contribute
            # zero probability, and their work is included in timing. With no
            # pipeline, causal schedules can safely stop at the query prefix.
            for tile in T.Pipelined(
                T.ceildiv(T.min((bx + 1) * block_m, sequence), block_n) if causal and stages == 0 else T.ceildiv(sequence, block_n),
                num_stages=stages,
            ):
                T.copy(K[bz, tile * block_n : (tile + 1) * block_n, by, :], k, coalesced_width=copy_width)
                T.clear(scores_mma)
                T.gemm(q, k, scores_mma, transpose_B=True, policy=qk)
                # A contiguous SIMT layout supports reductions even when QK's
                # MMA layout splits a reduction segment across warps.
                T.copy(scores_mma, scores_shared)
                T.copy(scores_shared, scores)
                for i, j in T.Parallel(block_m, block_n):
                    if causal:
                        scores[i, j] = T.if_then_else(
                            tile * block_n + j < sequence and tile * block_n + j <= bx * block_m + i, scores[i, j], -T.infinity("float32")
                        )
                    else:
                        scores[i, j] = T.if_then_else(tile * block_n + j < sequence, scores[i, j], -T.infinity("float32"))
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1)
                for i in T.Parallel(block_m):
                    maximum[i] = T.max(maximum[i], previous[i])
                    rescale[i] = T.exp2((previous[i] - maximum[i]) * scale)
                for i, j in T.Parallel(block_m, block_n):
                    scores[i, j] = T.exp2(scores[i, j] * scale - maximum[i] * scale)
                T.reduce_sum(scores, partial, dim=1)
                for i in T.Parallel(block_m):
                    total[i] = total[i] * rescale[i] + partial[i]
                # Shared probabilities allow the second GEMM to choose its own
                # partition. Layout conversion and synchronization are measured.
                T.copy(scores, probabilities, coalesced_width=copy_width)
                T.copy(rescale, rescale_shared)
                for i, j in T.Parallel(block_m, dim):
                    out[i, j] *= rescale_shared[i]
                T.copy(V[bz, tile * block_n : (tile + 1) * block_n, by, :], v, coalesced_width=copy_width)
                T.gemm(probabilities, v, out, policy=pv)
            T.copy(total, total_shared)
            for i, j in T.Parallel(block_m, dim):
                out[i, j] /= total_shared[i]
            T.copy(out, out_shared, coalesced_width=copy_width)
            T.copy(out_shared, O[bz, bx * block_m : (bx + 1) * block_m, by, :], coalesced_width=copy_width)

    return kernel

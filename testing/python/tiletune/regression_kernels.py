"""Fixed IR fixtures for compiler accounting regressions.

These programs preserve established regression expectations after their former
experiment implementations were retired. Experiment entry points never import
this module; the four final experiments use only their example kernels.
"""

import tilelang.language as T


def positive_integer(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _row_layout(block_rows, block_cols, threads, vector, row_threads):
    for name, value in dict(block_rows=block_rows, block_cols=block_cols, threads=threads, vector=vector, row_threads=row_threads).items():
        positive_integer(name, value)
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if row_threads > block_rows or threads % row_threads:
        raise ValueError("row thread groups must partition threads and fit block_rows")
    column_threads = threads // row_threads
    if block_cols < column_threads * vector:
        raise ValueError("column tile cannot supply the declared vector/thread layout")
    matrix = T.Fragment(
        (block_rows, block_cols),
        forward_fn=lambda i, j: (
            (i % row_threads) * column_threads + (j // vector) % column_threads,
            (i // row_threads) * (block_cols // column_threads) + (j // (vector * column_threads)) * vector + j % vector,
        ),
    )
    reduced = T.Fragment(
        (block_rows,), forward_fn=lambda i, r: ((i % row_threads) * column_threads + r, i // row_threads), replicate=column_threads
    )
    return matrix, reduced


def gemm_policy(name):
    policies = dict(square=T.GemmWarpPolicy.Square, full_row=T.GemmWarpPolicy.FullRow, full_col=T.GemmWarpPolicy.FullCol)
    if name not in policies:
        raise ValueError("warp policy must be square, full_row, or full_col")
    return policies[name]


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


def row_reduction_program(rows, columns, dtype, op, epsilon, block_rows, threads):
    """Retain the former full-row RMSNorm/sum graph for compiler regressions."""
    if op not in ("rmsnorm", "reduce_sum"):
        raise ValueError("row reduction fixture supports rmsnorm and reduce_sum")
    width = 1 << (columns - 1).bit_length()
    output_shape = (rows,) if op == "reduce_sum" else (rows, columns)

    @T.prim_func
    def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor(output_shape, dtype)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
            x = T.alloc_fragment((block_rows, width), "float32")
            values = T.alloc_fragment((block_rows, width), "float32")
            reduced = T.alloc_fragment((block_rows,), "float32")
            T.copy(X[bx * block_rows, 0], x)
            if op == "rmsnorm":
                for i, j in T.Parallel(block_rows, width):
                    values[i, j] = x[i, j] * x[i, j]
                T.reduce_sum(values, reduced, dim=1)
                for i, j in T.Parallel(block_rows, width):
                    x[i, j] = x[i, j] * T.rsqrt(reduced[i] / columns + epsilon)
            else:
                T.reduce_sum(x, reduced, dim=1)
            if op == "reduce_sum":
                T.copy(reduced, Y[bx * block_rows])
            else:
                T.copy(x, Y[bx * block_rows, 0])

    return kernel


def softmax_program(rows, columns, dtype, block_rows, block_cols, threads, vector, row_threads):
    (matrix_layout, reduced_layout) = _row_layout(block_rows, block_cols, threads, vector, row_threads)
    output_shape = (rows, columns)

    @T.prim_func
    def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor(output_shape, dtype)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
            x = T.alloc_fragment((block_rows, block_cols), "float32")
            values = T.alloc_fragment((block_rows, block_cols), "float32")
            partial = T.alloc_fragment((block_rows,), "float32")
            total = T.alloc_fragment((block_rows,), "float32")
            maximum = T.alloc_fragment((block_rows,), "float32")
            next_maximum = T.alloc_fragment((block_rows,), "float32")
            T.annotate_layout(
                {
                    x: matrix_layout,
                    values: matrix_layout,
                    partial: reduced_layout,
                    total: reduced_layout,
                    maximum: reduced_layout,
                    next_maximum: reduced_layout,
                }
            )
            T.clear(total)
            T.fill(maximum, -T.infinity("float32"))
            for tile in T.serial(T.ceildiv(columns, block_cols)):
                T.copy(X[bx * block_rows, tile * block_cols], x, coalesced_width=vector)
                for i, j in T.Parallel(block_rows, block_cols):
                    x[i, j] = T.if_then_else(tile * block_cols + j < columns, x[i, j], -T.infinity("float32"))
                T.reduce_max(x, next_maximum, dim=1)
                for i in T.Parallel(block_rows):
                    next_maximum[i] = T.max(maximum[i], next_maximum[i])
                for i, j in T.Parallel(block_rows, block_cols):
                    values[i, j] = T.exp(x[i, j] - next_maximum[i])
                T.reduce_sum(values, partial, dim=1)
                for i in T.Parallel(block_rows):
                    total[i] = total[i] * T.exp(maximum[i] - next_maximum[i]) + partial[i]
                    maximum[i] = next_maximum[i]
            for tile in T.serial(T.ceildiv(columns, block_cols)):
                T.copy(X[bx * block_rows, tile * block_cols], x, coalesced_width=vector)
                for i, j in T.Parallel(block_rows, block_cols):
                    values[i, j] = T.exp(x[i, j] - maximum[i]) / total[i]
                T.copy(values, Y[bx * block_rows, tile * block_cols], coalesced_width=vector)

    return kernel


def chunk_program(batch, heads, sequence, dk, dv, chunk, dtype, block_k, block_v, stages, threads, block_m, block_s, intra_stages):
    """Tile output rows and both reduction axes independently of chunk size."""
    for name, value in dict(block_m=block_m, block_k=block_k, block_v=block_v, block_s=block_s).items():
        if type(value) is not int or value <= 0 or value % 16:
            raise ValueError(f"{name} must be a positive multiple of 16")
    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be a positive integer")
    for name, value in dict(stages=stages, intra_stages=intra_stages).items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    chunks, row_tiles = sequence // chunk, (chunk + block_m - 1) // block_m
    qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
    ashape, hshape = (batch, heads, sequence, chunk), (batch, heads, chunks, dk, dv)
    scale = dk**-0.5

    @T.prim_func
    def kernel(
        Q: T.Tensor(qshape, dtype),
        V: T.Tensor(vshape, dtype),
        G: T.Tensor(qshape, "float32"),
        A: T.Tensor(ashape, dtype),
        H: T.Tensor(hshape, dtype),
        O: T.Tensor(vshape, dtype),
    ):
        with T.Kernel(T.ceildiv(dv, block_v), chunks * row_tiles, batch * heads, threads=threads) as (bv, br, bh):
            bc, row = br // row_tiles, (br % row_tiles) * block_m
            q = T.alloc_shared((block_m, block_k), dtype)
            g = T.alloc_shared((block_m, block_k), "float32")
            gq = T.alloc_shared((block_m, block_k), dtype)
            h = T.alloc_shared((block_k, block_v), dtype)
            a_raw = T.alloc_shared((block_m, block_s), dtype)
            a = T.alloc_shared((block_m, block_s), dtype)
            v = T.alloc_shared((block_s, block_v), dtype)
            out = T.alloc_fragment((block_m, block_v), "float32")
            T.clear(out)
            for kk in T.Pipelined(T.ceildiv(dk, block_k), num_stages=stages):
                T.copy(Q[bh // heads, bh % heads, bc * chunk + row, kk * block_k], q)
                T.copy(G[bh // heads, bh % heads, bc * chunk + row, kk * block_k], g)
                T.copy(H[bh // heads, bh % heads, bc, kk * block_k, bv * block_v], h)
                for i, j in T.Parallel(block_m, block_k):
                    gq[i, j] = q[i, j] * scale * T.exp2(g[i, j])
                T.gemm(gq, h, out)
            # Dynamic short prefixes can misalign pipeline prefetch/drain
            # indices. Pipelined schedules use a fixed full-chunk trip count;
            # future columns remain zero and their extra work is measured.
            # The serial schedule retains the shorter causal prefix.
            for ss in T.Pipelined(
                T.ceildiv(chunk, block_s) if intra_stages else T.ceildiv(T.min(row + block_m, chunk), block_s),
                num_stages=intra_stages,
            ):
                T.copy(A[bh // heads, bh % heads, bc * chunk + row, ss * block_s], a_raw)
                T.copy(V[bh // heads, bh % heads, bc * chunk + ss * block_s, bv * block_v], v)
                for i, j in T.Parallel(block_m, block_s):
                    a[i, j] = T.if_then_else(ss * block_s + j <= row + i and ss * block_s + j < chunk, a_raw[i, j], 0)
                T.gemm(a, v, out)
            # A partial row tile must not overwrite the next chunk's output.
            for i, j in T.Parallel(block_m, block_v):
                if row + i < chunk and bv * block_v + j < dv:
                    O[bh // heads, bh % heads, bc * chunk + row + i, bv * block_v + j] = out[i, j]

    return kernel


def recurrent_program(batch, heads, sequence, dk, dv, dtype, block_v, threads):
    qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
    scale = dk**-0.5

    @T.prim_func
    def kernel(
        Q: T.Tensor(qshape, dtype),
        K: T.Tensor(qshape, dtype),
        V: T.Tensor(vshape, dtype),
        G: T.Tensor(qshape, "float32"),
        Beta: T.Tensor((batch, heads, sequence), "float32"),
        O: T.Tensor(vshape, dtype),
        Final: T.Tensor((batch, heads, dk, dv), "float32"),
    ):
        with T.Kernel(T.ceildiv(dv, block_v), batch * heads, threads=threads) as (bv, bh):
            state = T.alloc_fragment((dk, block_v), "float32")
            products = T.alloc_fragment((dk, block_v), "float32")
            prediction = T.alloc_fragment((block_v,), "float32")
            delta = T.alloc_fragment((block_v,), "float32")
            T.clear(state)
            for t in T.serial(sequence):
                for i, j in T.Parallel(dk, block_v):
                    state[i, j] *= T.exp(G[bh // heads, bh % heads, t, i])
                    products[i, j] = state[i, j] * K[bh // heads, bh % heads, t, i]
                T.reduce_sum(products, prediction, dim=0)
                for j in T.Parallel(block_v):
                    delta[j] = (
                        T.if_then_else(bv * block_v + j < dv, V[bh // heads, bh % heads, t, bv * block_v + j], 0) - prediction[j]
                    ) * Beta[bh // heads, bh % heads, t]
                for i, j in T.Parallel(dk, block_v):
                    state[i, j] += K[bh // heads, bh % heads, t, i] * delta[j]
                    products[i, j] = state[i, j] * Q[bh // heads, bh % heads, t, i] * scale
                T.reduce_sum(products, prediction, dim=0)
                T.copy(prediction, O[bh // heads, bh % heads, t, bv * block_v])
            T.copy(state, Final[bh // heads, bh % heads, 0, bv * block_v])

    return kernel

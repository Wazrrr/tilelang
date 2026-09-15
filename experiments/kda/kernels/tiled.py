"""Tiled recurrent and chunk-output KDA implementations."""

import tilelang.language as T


def _kda_recurrent_tiled(batch, heads, sequence, dk, dv, dtype, block_v, threads, block_t, stages, unroll):
    """Stage several tokens per CTA while keeping the FP32 recurrence live.

    The token loop stays ordered. Unrolling changes code generation, and the
    pipeline overlaps input copies with computation across token tiles.
    """
    for name, value in dict(block_v=block_v, threads=threads, block_t=block_t, unroll=unroll).items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(stages) is not int or stages < 0:
        raise ValueError("stages must be a nonnegative integer")
    if block_t % unroll:
        raise ValueError("unroll must divide block_t")
    # Padding also supports key dimensions that are not powers of two.
    width = 1 << (dk - 1).bit_length()
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
            q = T.alloc_shared((block_t, width), dtype)
            k = T.alloc_shared((block_t, width), dtype)
            g = T.alloc_shared((block_t, width), "float32")
            v = T.alloc_shared((block_t, block_v), dtype)
            beta = T.alloc_shared((block_t,), "float32")
            state = T.alloc_fragment((width, block_v), "float32")
            products = T.alloc_fragment((width, block_v), "float32")
            prediction = T.alloc_fragment((block_v,), "float32")
            delta = T.alloc_fragment((block_v,), "float32")
            T.clear(state)
            for tile in T.Pipelined(T.ceildiv(sequence, block_t), num_stages=stages):
                T.copy(Q[bh // heads, bh % heads, tile * block_t, 0], q)
                T.copy(K[bh // heads, bh % heads, tile * block_t, 0], k)
                T.copy(G[bh // heads, bh % heads, tile * block_t, 0], g)
                T.copy(V[bh // heads, bh % heads, tile * block_t, bv * block_v], v)
                T.copy(Beta[bh // heads, bh % heads, tile * block_t], beta)
                for group in T.serial(block_t // unroll):
                    for u in T.unroll(unroll):
                        t = group * unroll + u
                        # Skipping padded tokens is essential: even zero inputs
                        # must not decay the final state after the last token.
                        if tile * block_t + t < sequence:
                            for i, j in T.Parallel(width, block_v):
                                state[i, j] *= T.exp(g[t, i])
                                products[i, j] = state[i, j] * k[t, i]
                            T.reduce_sum(products, prediction, dim=0)
                            for j in T.Parallel(block_v):
                                delta[j] = (v[t, j] - prediction[j]) * beta[t]
                            for i, j in T.Parallel(width, block_v):
                                state[i, j] += k[t, i] * delta[j]
                                products[i, j] = state[i, j] * q[t, i] * scale
                            T.reduce_sum(products, prediction, dim=0)
                            T.copy(prediction, O[bh // heads, bh % heads, tile * block_t + t, bv * block_v])
            T.copy(state, Final[bh // heads, bh % heads, 0, bv * block_v])

    return kernel


def _kda_chunk_tiled(batch, heads, sequence, dk, dv, chunk, dtype, block_k, block_v, stages, threads, block_m, block_s, intra_stages):
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

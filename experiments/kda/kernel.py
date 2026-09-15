"""Family-owned portable implementations and independent references."""

import torch
import tilelang.language as T
from experiments._kernel import KernelCase, _random
from .reference import recurrent_reference, chunk_reference
from .kernels.tiled import _kda_recurrent_tiled, _kda_chunk_tiled


def kda_recurrent_case(w):
    p, dtype = w.parameters, w.dtype
    batch, heads, sequence, dk, dv = (p[k] for k in ("batch", "heads", "sequence", "dim", "value_dim"))
    qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
    scale = dk**-0.5

    def build(block_v, threads, implementation="baseline", block_t=4, stages=0, unroll=1):
        if implementation == "tiled":
            return _kda_recurrent_tiled(batch, heads, sequence, dk, dv, dtype, block_v, threads, block_t, stages, unroll)
        if implementation != "baseline":
            raise ValueError("implementation must be baseline or tiled")
        if block_t != 4 or stages != 0 or unroll != 1:
            raise ValueError("token tiling parameters require implementation='tiled'")
        qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)

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

    def inputs(device, generator):
        q = _random(qshape, dtype, device, generator)
        k = torch.nn.functional.normalize(_random(qshape, "float32", device, generator), dim=-1).to(getattr(torch, dtype))
        v = _random(vshape, dtype, device, generator)
        g = -torch.rand(qshape, device=device, generator=generator) * 0.1
        beta = torch.rand((batch, heads, sequence), device=device, generator=generator)
        return [q, k, v, g, beta]

    return KernelCase(build, inputs, recurrent_reference(w), [5, 6])


def kda_chunk_case(w):
    p, dtype = w.parameters, w.dtype
    batch, heads, sequence, dk, dv, chunk = (p[k] for k in ("batch", "heads", "sequence", "dim", "value_dim", "chunk_size"))
    chunks = sequence // chunk
    qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
    ashape, hshape = (batch, heads, sequence, chunk), (batch, heads, chunks, dk, dv)
    scale = dk**-0.5

    def build(block_k, block_v, stages, threads, implementation="baseline", block_m=32, block_s=32, intra_stages=0):
        if implementation == "tiled":
            return _kda_chunk_tiled(
                batch, heads, sequence, dk, dv, chunk, dtype, block_k, block_v, stages, threads, block_m, block_s, intra_stages
            )
        if implementation != "baseline":
            raise ValueError("implementation must be baseline or tiled")
        if block_m != 32 or block_s != 32 or intra_stages != 0:
            raise ValueError("chunk tiling parameters require implementation='tiled'")
        qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
        ashape, hshape = (batch, heads, sequence, chunk), (batch, heads, chunks, dk, dv)

        @T.prim_func
        def kernel(
            Q: T.Tensor(qshape, dtype),
            V: T.Tensor(vshape, dtype),
            G: T.Tensor(qshape, "float32"),
            A: T.Tensor(ashape, dtype),
            H: T.Tensor(hshape, dtype),
            O: T.Tensor(vshape, dtype),
        ):
            with T.Kernel(T.ceildiv(dv, block_v), chunks, batch * heads, threads=threads) as (bv, bc, bh):
                q = T.alloc_shared((chunk, block_k), dtype)
                g = T.alloc_shared((chunk, block_k), "float32")
                gq = T.alloc_shared((chunk, block_k), dtype)
                h = T.alloc_shared((block_k, block_v), dtype)
                a = T.alloc_shared((chunk, chunk), dtype)
                v = T.alloc_shared((chunk, block_v), dtype)
                out = T.alloc_fragment((chunk, block_v), "float32")
                T.clear(out)
                for kk in T.Pipelined(T.ceildiv(dk, block_k), num_stages=stages):
                    T.copy(Q[bh // heads, bh % heads, bc * chunk, kk * block_k], q)
                    T.copy(G[bh // heads, bh % heads, bc * chunk, kk * block_k], g)
                    T.copy(H[bh // heads, bh % heads, bc, kk * block_k, bv * block_v], h)
                    for i, j in T.Parallel(chunk, block_k):
                        gq[i, j] = q[i, j] * scale * T.exp2(g[i, j])
                    T.gemm(gq, h, out)
                T.copy(A[bh // heads, bh % heads, bc * chunk, 0], a)
                T.copy(V[bh // heads, bh % heads, bc * chunk, bv * block_v], v)
                for i, j in T.Parallel(chunk, chunk):
                    a[i, j] = T.if_then_else(j <= i, a[i, j], 0)
                T.gemm(a, v, out)
                T.copy(out, O[bh // heads, bh % heads, bc * chunk, bv * block_v])

        return kernel

    def inputs(device, generator):
        return [
            _random(qshape, dtype, device, generator),
            _random(vshape, dtype, device, generator),
            _random(qshape, "float32", device, generator),
            _random(ashape, dtype, device, generator),
            _random(hshape, dtype, device, generator),
        ]

    return KernelCase(build, inputs, chunk_reference(w), [5])


def make_case(workload):
    return kda_recurrent_case(workload) if workload.op == "kda_recurrent" else kda_chunk_case(workload)

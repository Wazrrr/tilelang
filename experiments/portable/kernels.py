"""Portable GPU workload implementations and independent PyTorch references.

Builders elaborate IR only. Input allocation and reference execution are explicit
operations on the selected device. KDA includes a complete recurrent forward
baseline and the chunk-output stage; neither is labeled a full optimized chunked
KDA pipeline. Gates in the recurrence are natural logarithms; chunk gates use
base-two cumulative values, matching examples/kda/chunk_o.py.
"""

from dataclasses import dataclass, field
from threading import Lock

import torch
import tilelang.language as T

_ATTENTION_LOCK = Lock()


def _attention_program(**kwargs):
    from examples.flash_attention.example_mha_fwd_bshd import flashattn

    with _ATTENTION_LOCK:
        return flashattn.jit_impl.get_tir(**kwargs)


@dataclass
class KernelCase:
    build: object
    inputs: object
    reference: object
    out_idx: list[int]
    pass_configs: dict = field(default_factory=dict)
    rtol: float = 0.02
    atol: float = 0.02

    def check(self, actuals, references):
        """Check both elementwise errors and relative signal error.

        An absolute tolerance alone can accept all-zero softmax/attention output
        when sequence lengths are large. The norm check prevents that failure.
        """
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        for actual, reference in zip(actuals, references):
            torch.testing.assert_close(actual, reference, rtol=self.rtol, atol=self.atol)
            error = torch.linalg.vector_norm(actual.float() - reference.float())
            signal = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            if not torch.isfinite(error) or error / signal > self.rtol:
                raise AssertionError(f"relative output norm error {(error / signal).item()} exceeds {self.rtol}")


def _random(shape, dtype, device, generator):
    # Generate in FP32 so FP8 uses the same seeded input distribution.
    return (torch.rand(shape, device=device, generator=generator) - 0.5).to(getattr(torch, dtype))


def gemm_case(w):
    p, dtype = w.parameters, w.dtype
    batch, m, n, k = p.get("batch", 1), p["m"], p["n"], p["k"]
    ta, tb, epilogue = p.get("transpose_a", False), p.get("transpose_b", False), p.get("epilogue", "none")
    ashape, bshape = (batch, k, m) if ta else (batch, m, k), (batch, n, k) if tb else (batch, k, n)
    output_dtype = "float16" if dtype.startswith("float8") else dtype

    def build(block_m, block_n, block_k, stages, threads):
        ashape = (batch, k, m) if ta else (batch, m, k)
        bshape = (batch, n, k) if tb else (batch, k, n)

        @T.prim_func
        def kernel(
            A: T.Tensor(ashape, dtype),
            B: T.Tensor(bshape, dtype),
            Bias: T.Tensor((n,), output_dtype),
            C: T.Tensor((batch, m, n), output_dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), batch, threads=threads) as (bx, by, bz):
                a = T.alloc_shared((block_k, block_m) if ta else (block_m, block_k), dtype)
                b = T.alloc_shared((block_n, block_k) if tb else (block_k, block_n), dtype)
                c = T.alloc_fragment((block_m, block_n), "float32")
                T.clear(c)
                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=stages):
                    if ta:
                        T.copy(A[bz, kk * block_k, by * block_m], a)
                    else:
                        T.copy(A[bz, by * block_m, kk * block_k], a)
                    if tb:
                        T.copy(B[bz, bx * block_n, kk * block_k], b)
                    else:
                        T.copy(B[bz, kk * block_k, bx * block_n], b)
                    T.gemm(a, b, c, transpose_A=ta, transpose_B=tb)
                if epilogue != "none":
                    for i, j in T.Parallel(block_m, block_n):
                        c[i, j] += Bias[bx * block_n + j]
                    if epilogue == "bias_relu":
                        for i, j in T.Parallel(block_m, block_n):
                            c[i, j] = T.max(c[i, j], 0)
                T.copy(c, C[bz, by * block_m, bx * block_n])

        return kernel

    def inputs(device, generator):
        return [
            _random(ashape, dtype, device, generator),
            _random(bshape, dtype, device, generator),
            _random((n,), output_dtype, device, generator),
        ]

    def reference(a, b, bias):
        a, b = a.float(), b.float()
        result = (a.transpose(-1, -2) if ta else a) @ (b.transpose(-1, -2) if tb else b)
        if epilogue != "none":
            result = result + bias.float()
        if epilogue == "bias_relu":
            result = result.relu()
        return result.to(getattr(torch, output_dtype))

    return KernelCase(build, inputs, reference, [3])


def attention_case(w):
    # Reuse the existing FlashAttention algorithm, including its stable online
    # softmax. The eager builder is mutable, so serialize elaboration only.
    from examples.flash_attention.example_mha_tiletune import reference_attention

    p = w.parameters
    batch, heads, sequence, dim = (p[key] for key in ("batch", "heads", "sequence", "dim"))
    causal = p.get("causal", False)
    dtype = w.dtype

    def build(block_M, block_N, num_stages, threads):
        return _attention_program(
            batch=batch,
            heads=heads,
            seq_len=sequence,
            dim=dim,
            is_causal=causal,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            dtype=dtype,
        )

    def inputs(device, generator):
        return [_random((batch, sequence, heads, dim), w.dtype, device, generator) for _ in range(3)]

    return KernelCase(build, inputs, lambda q, k, v: reference_attention(q, k, v, causal).to(q.dtype), [3], {"tl.enable_fast_math": True})


def row_case(w):
    rows, columns = w.parameters["rows"], w.parameters["columns"]
    width = 1 << (columns - 1).bit_length()
    dtype, op = w.dtype, w.op
    epsilon = w.parameters.get("epsilon", 1e-6)

    def build(block_rows, threads):
        output_shape = (rows,) if op == "reduce_sum" else (rows, columns)

        @T.prim_func
        def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor(output_shape, dtype)):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
                x = T.alloc_fragment((block_rows, width), "float32")
                values = T.alloc_fragment((block_rows, width), "float32")
                reduced = T.alloc_fragment((block_rows,), "float32")
                T.copy(X[bx * block_rows, 0], x)
                if op == "softmax":
                    for i, j in T.Parallel(block_rows, width):
                        x[i, j] = T.if_then_else(j < columns, x[i, j], -T.infinity("float32"))
                    T.reduce_max(x, reduced, dim=1)
                    for i, j in T.Parallel(block_rows, width):
                        values[i, j] = T.exp(x[i, j] - reduced[i])
                    T.reduce_sum(values, reduced, dim=1)
                    for i, j in T.Parallel(block_rows, width):
                        x[i, j] = values[i, j] / reduced[i]
                elif op == "rmsnorm":
                    for i, j in T.Parallel(block_rows, width):
                        values[i, j] = x[i, j] * x[i, j]
                    T.reduce_sum(values, reduced, dim=1)
                    for i, j in T.Parallel(block_rows, width):
                        x[i, j] = x[i, j] * T.rsqrt(reduced[i] / columns + epsilon)
                elif op == "reduce_sum":
                    T.reduce_sum(x, reduced, dim=1)
                else:
                    for i, j in T.Parallel(block_rows, width):
                        x[i, j] = T.max(x[i, j] * 2 + 1, 0)
                if op == "reduce_sum":
                    T.copy(reduced, Y[bx * block_rows])
                else:
                    T.copy(x, Y[bx * block_rows, 0])

        return kernel

    def reference(x):
        xf = x.float()
        if op == "softmax":
            result = xf.softmax(dim=-1)
        elif op == "rmsnorm":
            result = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + epsilon)
        elif op == "reduce_sum":
            result = xf.sum(dim=-1)
        else:
            result = (2 * xf + 1).relu()
        return result.to(x.dtype)

    return KernelCase(
        build,
        lambda device, generator: [_random((rows, columns), dtype, device, generator)],
        reference,
        [1],
        atol=1e-5 if op == "softmax" else 0.02,
    )


def kda_recurrent_case(w):
    p, dtype = w.parameters, w.dtype
    batch, heads, sequence, dk, dv = (p[k] for k in ("batch", "heads", "sequence", "dim", "value_dim"))
    qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
    scale = dk**-0.5

    def build(block_v, threads):
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

    def reference(q, k, v, g, beta):
        state = torch.zeros((batch, heads, dk, dv), device=q.device, dtype=torch.float32)
        out = []
        for t in range(sequence):
            state = state * g[:, :, t].exp().unsqueeze(-1)
            kt = k[:, :, t].float()
            residual = (v[:, :, t].float() - torch.einsum("bhd,bhdv->bhv", kt, state)) * beta[:, :, t, None]
            state = state + kt.unsqueeze(-1) * residual.unsqueeze(-2)
            out.append(torch.einsum("bhd,bhdv->bhv", q[:, :, t].float() * scale, state))
        return torch.stack(out, dim=2).to(q.dtype), state

    return KernelCase(build, inputs, reference, [5, 6])


def kda_chunk_case(w):
    p, dtype = w.parameters, w.dtype
    batch, heads, sequence, dk, dv, chunk = (p[k] for k in ("batch", "heads", "sequence", "dim", "value_dim", "chunk_size"))
    chunks = sequence // chunk
    qshape, vshape = (batch, heads, sequence, dk), (batch, heads, sequence, dv)
    ashape, hshape = (batch, heads, sequence, chunk), (batch, heads, chunks, dk, dv)
    scale = dk**-0.5

    def build(block_k, block_v, stages, threads):
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

    def reference(q, v, g, a, h):
        # Match the materialized gated-query dtype before the matrix operation.
        gq = (q.float() * scale * g.exp2()).to(q.dtype).float().reshape(batch, heads, chunks, chunk, dk)
        local = a.float().reshape(batch, heads, chunks, chunk, chunk).tril()
        result = gq @ h.float() + local @ v.float().reshape(batch, heads, chunks, chunk, dv)
        return result.reshape(vshape).to(q.dtype)

    return KernelCase(build, inputs, reference, [5])


def make_case(workload):
    if workload.op == "gemm":
        return gemm_case(workload)
    if workload.op == "attention":
        return attention_case(workload)
    if workload.op == "kda_recurrent":
        return kda_recurrent_case(workload)
    if workload.op == "kda_chunk_o":
        return kda_chunk_case(workload)
    return row_case(workload)

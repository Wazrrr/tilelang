"""Independent chunk-output KDA reference, including intermediate casts."""


def chunk_reference(w):
    batch, heads, sequence, dk, dv, chunk = (w.parameters[k] for k in ("batch", "heads", "sequence", "dim", "value_dim", "chunk_size"))
    chunks = sequence // chunk
    scale = dk**-0.5

    def reference(q, v, g, a, h):
        # The example stores scaled Q in input dtype, then materializes gated Q
        # in that dtype before GEMM. Preserve both rounding points.
        scaled_q = (q.float() * scale).to(q.dtype)
        gq = (scaled_q.float() * g.exp2()).to(q.dtype).float()
        gq = gq.permute(0, 2, 1, 3).reshape(batch, heads, chunks, chunk, dk)
        local = a.float().permute(0, 2, 1, 3).reshape(batch, heads, chunks, chunk, chunk).tril()
        values = v.float().permute(0, 2, 1, 3).reshape(batch, heads, chunks, chunk, dv)
        result = gq @ h.float().permute(0, 2, 1, 3, 4) + local @ values
        return result.reshape(batch, heads, sequence, dv).permute(0, 2, 1, 3).contiguous().to(q.dtype)

    return reference

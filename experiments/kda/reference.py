"""Independent recurrent and chunk-output KDA references."""

import torch


def recurrent_reference(w):
    batch, heads, sequence, dk, dv = (w.parameters[k] for k in ("batch", "heads", "sequence", "dim", "value_dim"))
    scale = dk**-0.5

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

    return reference


def chunk_reference(w):
    batch, heads, sequence, dk, dv, chunk = (w.parameters[k] for k in ("batch", "heads", "sequence", "dim", "value_dim", "chunk_size"))
    chunks = sequence // chunk
    vshape = (batch, heads, sequence, dv)
    scale = dk**-0.5

    def reference(q, v, g, a, h):
        # Match the materialized gated-query dtype before the matrix operation.
        gq = (q.float() * scale * g.exp2()).to(q.dtype).float().reshape(batch, heads, chunks, chunk, dk)
        local = a.float().reshape(batch, heads, chunks, chunk, chunk).tril()
        result = gq @ h.float() + local @ v.float().reshape(batch, heads, chunks, chunk, dv)
        return result.reshape(vshape).to(q.dtype)

    return reference

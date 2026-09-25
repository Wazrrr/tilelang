"""Independent reference for token-parallel KDA intra-chunk coefficients."""


def chunk_intra_reference(w):
    batch, heads, sequence, dim, chunk, sub_chunk = (
        w.parameters[key] for key in ("batch", "heads", "sequence", "dim", "chunk_size", "sub_chunk_size")
    )
    chunks = sequence // chunk
    sub_chunks = chunk // sub_chunk
    scale = dim**-0.5

    def reference(q, k, gates, beta):
        import torch

        grouped_shape = (batch, chunks, sub_chunks, sub_chunk, heads, dim)
        q_groups = q.float().reshape(grouped_shape)
        k_groups = k.float().reshape(grouped_shape)
        gate_groups = gates.float().reshape(grouped_shape)
        beta_groups = beta.float().reshape(batch, chunks, sub_chunks, sub_chunk, heads)

        # A per-sub-chunk offset keeps the algebraic exp2 factorization stable:
        # exp2(g_i - g_j) = exp2(g_i - offset) * exp2(offset - g_j).
        offset = gate_groups[:, :, :, :1]
        gated_q = q_groups * torch.exp2(gate_groups - offset) * scale
        gated_k = k_groups * torch.exp2(offset - gate_groups)
        beta_k = k_groups * beta_groups[..., None] * torch.exp2(gate_groups - offset)
        aqk_local = torch.einsum("bcuqhd,bcukhd->bcuqkh", gated_q, gated_k)
        akk_local = torch.einsum("bcuqhd,bcukhd->bcuqkh", beta_k, gated_k)

        lower = torch.ones((sub_chunk, sub_chunk), dtype=torch.bool, device=q.device).tril()
        strict_lower = lower.logical_xor(torch.eye(sub_chunk, dtype=torch.bool, device=q.device))
        aqk_local = aqk_local.masked_fill(~lower[None, None, None, :, :, None], 0)
        akk_local = akk_local.masked_fill(~strict_lower[None, None, None, :, :, None], 0)

        aqk = torch.zeros((batch, chunks, chunk, heads, chunk), dtype=torch.float32, device=q.device)
        aqk_local = aqk_local.permute(0, 1, 2, 3, 5, 4)
        for index in range(sub_chunks):
            token_slice = slice(index * sub_chunk, (index + 1) * sub_chunk)
            aqk[:, :, token_slice, :, token_slice] = aqk_local[:, :, index]
        akk = akk_local.permute(0, 1, 2, 3, 5, 4)
        return (
            aqk.reshape(batch, sequence, heads, chunk).to(q.dtype),
            akk.reshape(batch, sequence, heads, sub_chunk).to(q.dtype),
        )

    return reference

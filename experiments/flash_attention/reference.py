"""FP32 attention reference and numerical validation."""

import torch


def reference_attention(q, k, v, causal):
    """FP32 reference, chunked over queries to bound temporary memory."""
    torch.backends.cuda.matmul.allow_tf32 = False
    qh, kh, vh = [x.permute(0, 2, 1, 3).float() for x in (q, k, v)]
    output = torch.empty_like(qh)
    sequence, dim = q.shape[1], q.shape[-1]
    keys = torch.arange(sequence, device=q.device)
    for start in range(0, sequence, 256):
        end = min(sequence, start + 256)
        scores = (qh[:, :, start:end] @ kh.transpose(-1, -2)) * dim**-0.5
        if causal:
            queries = torch.arange(start, end, device=q.device)
            scores.masked_fill_(keys[None, :] > queries[:, None], -float("inf"))
        output[:, :, start:end] = scores.softmax(-1) @ vh
    return output.permute(0, 2, 1, 3).contiguous()

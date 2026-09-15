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


def check_output(output, reference, rtol=0.02, atol=0.02):
    actual = output.float()
    error = (actual - reference).abs()
    mismatch = error > atol + rtol * reference.abs()
    result = dict(
        max_abs_error=error.max().item(),
        rms_error=error.square().mean().sqrt().item(),
        mismatched_elements=mismatch.sum().item(),
        finite=bool(torch.isfinite(actual).all().item()),
        rtol=rtol,
        atol=atol,
    )
    if not result["finite"] or result["mismatched_elements"]:
        raise AssertionError(f"Attention correctness failed: {result}")
    return result


def check_accuracy(actuals, refs):
    check_output(actuals[0], refs[0])


reference = reference_attention

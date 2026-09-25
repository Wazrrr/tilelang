"""Independent FP32 products for each contiguous group of A rows."""

import torch


def reference(a, b, batch_sizes, transpose_b=False):
    outputs = []
    start = 0
    for group, size in enumerate(batch_sizes):
        weight = b[group].float()
        if transpose_b:
            weight = weight.T
        outputs.append(a[start : start + size].float() @ weight)
        start += size
    return torch.cat(outputs, dim=0).to(a.dtype)

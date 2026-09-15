"""Numerical references for normalization, reduction and elementwise kernels."""

import torch


def make_reference(w):
    if w.op == "softmax":
        from experiments.softmax.reference import reference

        return reference
    op = w.op
    epsilon = w.parameters.get("epsilon", 1e-06)

    def reference(x):
        xf = x.float()
        if op == "rmsnorm":
            result = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + epsilon)
        elif op == "reduce_sum":
            result = xf.sum(dim=-1)
        else:
            result = (2 * xf + 1).relu()
        return result.to(x.dtype)

    return reference

"""Mathematical references for the GEMM implementations."""

import torch


def reference(a, b):
    return a @ b.T


def suite_reference(w):
    p = w.parameters
    ta, tb, epilogue = p.get("transpose_a", False), p.get("transpose_b", False), p.get("epilogue", "none")
    output_dtype = "float16" if w.dtype.startswith("float8") else w.dtype

    def reference(a, b, bias):
        a, b = a.float(), b.float()
        result = (a.transpose(-1, -2) if ta else a) @ (b.transpose(-1, -2) if tb else b)
        if epilogue != "none":
            result = result + bias.float()
        if epilogue == "bias_relu":
            result = result.relu()
        return result.to(getattr(torch, output_dtype))

    return reference

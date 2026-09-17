"""Independent FP32 matmul followed by the example's FP8 output rounding."""


def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)

"""Independent FP32 reference with the original example's FP8 output."""


def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)

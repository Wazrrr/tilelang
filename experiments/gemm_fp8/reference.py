"""Independent FP32 accumulation followed by the example's FP8 output cast."""


def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)

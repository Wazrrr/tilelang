"""Independent FP32-accumulating reference for FP8 C = A @ B.T."""


def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)

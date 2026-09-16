"""Independent FP32 reference for C = A @ B.T."""


def reference(a, b):
    return (a.float() @ b.float().T).to(a.dtype)

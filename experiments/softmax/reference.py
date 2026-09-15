"""FP32 softmax reference."""


def reference(x):
    return x.float().softmax(dim=-1).to(x.dtype)

"""Independent FP32 reference for the fixed Hopper block-scale layout."""


def reference(a, b, scale_a, scale_b):
    k = a.shape[1]
    a_dequant = a.float() * scale_a.repeat_interleave(128, dim=1)[:, :k]
    b_scale_rows = scale_b.repeat_interleave(128, dim=0)[: b.shape[0]]
    b_dequant = b.float() * b_scale_rows.repeat_interleave(128, dim=1)[:, :k]
    return (a_dequant @ b_dequant.T).bfloat16()

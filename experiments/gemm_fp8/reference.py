"""Independent FP32 dequantization reference for the common scale layout."""


def reference(a, b, scale_a, scale_b):
    a_dequant = a.float() * scale_a.repeat_interleave(128, dim=1)
    b_dequant = b.float() * scale_b.repeat_interleave(128, dim=1)
    return (a_dequant @ b_dequant.T).bfloat16()

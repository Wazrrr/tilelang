"""Grouped MXFP8 GEMMs for common MoE decode, prefill, and projections."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    shapes = (
        ("decode", [1, 2, 4, 8] if holdout else [1, 1, 2, 4], 2048, 7168, False),
        ("prefill", [16, 32, 48, 64] if holdout else [8, 16, 24, 32], 2048, 7168, False),
        ("aligned", [64, 128, 256] if holdout else [32, 64, 128], 2048, 7168, False),
        ("down_aligned", [64, 128, 256] if holdout else [32, 64, 128], 7168, 2048, True),
        ("ragged", [63, 77, 111, 280] if holdout else [31, 47, 81, 129], 7168, 2048, True),
    )
    return [
        Workload(
            "grouped_gemm_" + role,
            "grouped_gemm",
            dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b),
            dtype="float8_e4m3fn",
            config_space="expanded",
        )
        for role, sizes, n, k, transpose_b in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "grouped_gemm_" + split,
            "grouped_gemm",
            dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b),
            dtype="float8_e4m3fn",
            config_space="expanded",
        )
        for split, sizes, n, k, transpose_b in (
            ("train_a", [16, 32, 64], 2048, 7168, False),
            ("train_b", [15, 31, 65], 7168, 2048, True),
            ("validation", [24, 40, 72], 2048, 7168, False),
        )
    ]

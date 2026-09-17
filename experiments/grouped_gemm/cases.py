"""Grouped GEMMs for common MoE decode, prefill, and up/down projections."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    shapes = (
        ("decode", [1, 2, 4, 8] if holdout else [1, 1, 2, 4], 2048, 7168, False),
        ("prefill", [32] * 8 if holdout else [16] * 8, 2048, 7168, False),
        ("aligned", [128] * 4 if holdout else [64] * 4, 2048, 7168, False),
        ("down_aligned", [256] * 3 if holdout else [128] * 3, 7168, 2048, True),
        ("ragged", [63, 77, 111, 280] if holdout else [31, 47, 81, 129], 7168, 2048, True),
    )
    return [
        Workload(
            "grouped_gemm_" + role,
            "grouped_gemm",
            dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b),
            dtype="bfloat16",
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
            dtype="bfloat16",
            config_space="expanded",
        )
        for split, sizes, n, k, transpose_b in (
            ("train_a", [16] * 6, 2048, 7168, False),
            ("train_b", [15, 31, 65], 7168, 2048, True),
            ("validation", [24, 40, 72, 104, 136], 2048, 7168, False),
        )
    ]

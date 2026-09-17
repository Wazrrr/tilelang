"""FP8 LLM projection and feed-forward GEMMs."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    tokens = (128, 1024, 4096) if holdout else (64, 512, 2048)
    shapes = (
        ("decode_e4m3", tokens[0], 4096, 4096, "float8_e4m3fn"),
        ("prefill_e4m3", tokens[1], 4096, 4096, "float8_e4m3fn"),
        ("ffn_down_e4m3", tokens[1], 4096, 14336, "float8_e4m3fn"),
        ("e4m3", tokens[2], 4096, 4096, "float8_e4m3fn"),
        ("e5m2", tokens[2], 14336, 4096, "float8_e5m2"),
    )
    return [
        Workload(
            "gemm_fp8_" + role,
            "gemm_fp8",
            dict(m=m, n=n, k=k, transpose_b=True),
            dtype=dtype,
            config_space="expanded",
        )
        for role, m, n, k, dtype in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "gemm_fp8_" + split,
            "gemm_fp8",
            dict(m=m, n=n, k=k, transpose_b=True),
            dtype=dtype,
            config_space="expanded",
        )
        for split, (m, n, k), dtype in zip(
            ("train_a", "train_b", "validation"),
            ((32, 4096, 4096), (512, 14336, 4096), (2048, 4096, 4096)),
            ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fn"),
        )
    ]

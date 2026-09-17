"""FP8 LLM projection and feed-forward GEMMs."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    tokens = (256, 1024, 4096) if holdout else (64, 512, 2048)
    shapes = (
        ("decode", tokens[0], 4096, 4096),
        ("prefill", tokens[1], 4096, 4096),
        ("ffn_down", tokens[1], 4096, 14336),
        ("square", tokens[2], 4096, 4096),
        ("square_large", tokens[2], 14336, 4096),
    )
    return [
        Workload(
            "gemm_fp8_" + role,
            "gemm_fp8",
            dict(m=m, n=n, k=k, transpose_b=True),
            dtype="float8_e4m3fn",
            config_space="expanded",
        )
        for role, m, n, k in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "gemm_fp8_" + split,
            "gemm_fp8",
            dict(m=m, n=n, k=k, transpose_b=True),
            dtype="float8_e4m3fn",
            config_space="expanded",
        )
        for split, (m, n, k) in zip(
            ("train_a", "train_b", "validation"),
            ((768, 4096, 4096), (512, 14336, 4096), (3072, 4096, 4096)),
        )
    ]

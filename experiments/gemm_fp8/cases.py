"""FP8 GEMM cases covering both supported input encodings."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sizes = (4096, 8192) if holdout else (1024, 2048)
    return [
        Workload(
            "gemm_fp8_" + role,
            "gemm_fp8",
            dict(m=size, n=size, k=size, transpose_b=True),
            dtype=dtype,
            config_space="expanded",
        )
        for role, size, dtype in zip(("e4m3", "e5m2"), sizes, ("float8_e4m3fn", "float8_e5m2"))
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
            ((512, 512, 512), (1024, 256, 768), (384, 768, 512)),
            ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fn"),
        )
    ]

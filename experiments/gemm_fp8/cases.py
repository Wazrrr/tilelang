"""FP8 GEMM cases with disjoint training, validation, and final shapes."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sizes = (4096, 8192) if holdout else (1024, 2048)
    return [
        Workload("gemm_fp8_" + role, "gemm_fp8", dict(m=size, n=size, k=size, transpose_b=True), dtype="float8_e4m3fn")
        for role, size in zip(("square", "square_large"), sizes)
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload("gemm_fp8_" + role, "gemm_fp8", dict(m=m, n=n, k=k, transpose_b=True), dtype="float8_e4m3fn")
        for role, (m, n, k) in zip(("train_a", "train_b", "validation"), ((512, 512, 512), (1024, 256, 768), (384, 768, 512)))
    ]

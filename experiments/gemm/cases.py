"""Regular square FP16 GEMM cases and independent final holdouts."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sizes = (4096, 8192) if holdout else (1024, 2048)
    return [
        Workload("gemm_" + role, "gemm", dict(m=size, n=size, k=size), config_space="large")
        for role, size in zip(("square", "square_large"), sizes)
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload("gemm_" + split, "gemm", dict(m=m, n=n, k=k), config_space="large")
        for split, (m, n, k) in zip(("train_a", "train_b", "validation"), ((512, 512, 512), (1024, 256, 768), (384, 768, 512)))
    ]

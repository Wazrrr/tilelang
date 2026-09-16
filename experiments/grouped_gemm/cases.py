"""Aligned and ragged grouped GEMMs with independent model and final shapes."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    shapes = ((8192, 8192), (4096, 8192)) if holdout else ((512, 512), (768, 512))
    return [
        Workload(
            "grouped_gemm_" + role,
            "grouped_gemm",
            dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b),
            config_space="expanded",
        )
        for (role, sizes, transpose_b), (n, k) in zip((("aligned", [64, 128, 256], False), ("ragged", [63, 77, 111, 280], True)), shapes)
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "grouped_gemm_" + split,
            "grouped_gemm",
            dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b),
            config_space="expanded",
        )
        for split, sizes, n, k, transpose_b in (
            ("train_a", [32, 96], 256, 256, False),
            ("train_b", [47, 81, 129], 384, 256, True),
            ("validation", [65, 127], 256, 384, False),
        )
    ]

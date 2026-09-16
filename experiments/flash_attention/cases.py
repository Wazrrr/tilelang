"""Noncausal and causal attention cases with distinct sequence/head dimensions."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sequence = (768, 1152) if holdout else (512, 640)
    return [
        Workload("attention_" + role, "attention", dict(batch=1, heads=4, sequence=n, dim=d, causal=causal), config_space="expanded")
        for role, n, d, causal in (("noncausal", sequence[0], 64, False), ("causal", sequence[1], 128, True))
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload("attention_" + split, "attention", dict(batch=1, heads=2, sequence=n, dim=d, causal=c), config_space="expanded")
        for split, (n, d, c) in zip(("train_a", "train_b", "validation"), ((256, 64, False), (384, 128, True), (448, 64, True)))
    ]

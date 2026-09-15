"""Aligned and irregular FP16 softmax cases."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    shapes = ((1536, 4096), (1031, 1537)) if holdout else ((512, 1024), (519, 769))
    return [
        Workload("softmax_" + role, "softmax", dict(rows=r, columns=c), config_space="large")
        for role, (r, c) in zip(("aligned", "irregular"), shapes)
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload("softmax_" + split, "softmax", dict(rows=r, columns=c), config_space="large")
        for split, (r, c) in zip(("train_a", "train_b", "validation"), ((256, 512), (384, 1536), (263, 1023)))
    ]

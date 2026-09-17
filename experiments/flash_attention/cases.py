"""Noncausal and causal prefill attention at common LLM dimensions."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sequences = (512, 2048, 4096, 8192) if holdout else (256, 1024, 2048, 4096)
    shapes = (
        ("short_causal", sequences[0], True),
        ("medium_causal", sequences[1], True),
        ("noncausal", sequences[2], False),
        ("causal", sequences[2], True),
        ("long_causal", sequences[3], True),
    )
    return [
        Workload(
            "attention_" + role,
            "attention",
            dict(batch=1, heads=32, sequence=n, dim=128, causal=causal),
            config_space="expanded",
        )
        for role, n, causal in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "attention_" + split,
            "attention",
            dict(batch=1, heads=16, sequence=n, dim=128, causal=causal),
            config_space="expanded",
        )
        for split, (n, causal) in zip(
            ("train_a", "train_b", "validation"),
            ((512, False), (1024, True), (2048, True)),
        )
    ]

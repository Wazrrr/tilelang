"""Noncausal and causal prefill attention at common LLM dimensions."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sequences = (512, 2048, 4096, 8192) if holdout else (256, 1024, 2048, 4096)
    shapes = (
        ("short_causal", 1, 32, sequences[0], 64, True),
        ("batched_causal", 2, 16, sequences[1], 64, True),
        ("noncausal", 1, 32, sequences[2], 128, False),
        ("causal", 1, 32, sequences[2], 128, True),
        ("long_causal", 1, 16, sequences[3], 128, True),
    )
    return [
        Workload(
            "attention_" + role,
            "attention",
            dict(batch=batch, heads=heads, sequence=n, dim=dim, causal=causal),
            dtype="bfloat16",
            config_space="expanded",
        )
        for role, batch, heads, n, dim, causal in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    shapes = (
        ("train_a", 1, 16, 512, 64, False),
        ("train_b", 2, 8, 1024, 128, True),
        ("validation", 1, 24, 2048, 64, True),
    )
    return [
        Workload(
            "attention_" + split,
            "attention",
            dict(batch=batch, heads=heads, sequence=n, dim=dim, causal=causal),
            dtype="bfloat16",
            config_space="expanded",
        )
        for split, batch, heads, n, dim, causal in shapes
    ]

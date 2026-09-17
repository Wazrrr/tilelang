"""Chunk-output KDA at the serving dimensions used by the example family."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sequences = (2048, 4096, 8192, 16384) if holdout else (1024, 2048, 4096, 8192)
    shapes = (
        ("short", 1, 32, sequences[0]),
        ("medium", 1, 64, sequences[1]),
        ("regular", 1, 32, sequences[2]),
        ("batched", 2, 32, sequences[1]),
        ("long", 1, 64, sequences[3]),
    )
    return [
        Workload(
            "kda_chunk_" + role,
            "kda_chunk_o",
            dict(batch=batch, heads=heads, sequence=sequence, dim=128, value_dim=128, chunk_size=64),
            dtype="bfloat16",
            config_space="expanded",
        )
        for role, batch, heads, sequence in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    shapes = (
        ("train_a", 1, 16, 1024),
        ("train_b", 2, 16, 2048),
        ("validation", 1, 48, 4096),
    )
    return [
        Workload(
            "kda_chunk_" + split,
            "kda_chunk_o",
            dict(batch=batch, heads=heads, sequence=sequence, dim=128, value_dim=128, chunk_size=64),
            dtype="bfloat16",
            config_space="expanded",
        )
        for split, batch, heads, sequence in shapes
    ]

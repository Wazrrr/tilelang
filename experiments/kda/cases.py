"""Chunk-output KDA at the serving dimensions used by the example family."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    sequences = (2048, 4096, 8192, 16384) if holdout else (1024, 2048, 4096, 8192)
    shapes = (
        ("short", 1, sequences[0]),
        ("medium", 1, sequences[1]),
        ("regular", 1, sequences[2]),
        ("batched", 2, sequences[1]),
        ("long", 1, sequences[3]),
    )
    return [
        Workload(
            "kda_chunk_" + role,
            "kda_chunk_o",
            dict(batch=batch, heads=64, sequence=sequence, dim=128, value_dim=128, chunk_size=64),
            config_space="expanded",
        )
        for role, batch, sequence in shapes
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "kda_chunk_" + split,
            "kda_chunk_o",
            dict(batch=1, heads=32, sequence=sequence, dim=128, value_dim=128, chunk_size=64),
            config_space="expanded",
        )
        for split, sequence in zip(("train_a", "train_b", "validation"), (1024, 2048, 4096))
    ]

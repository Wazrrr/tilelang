"""Chunk-output KDA only; sequence lengths always contain complete chunks."""


def cases(holdout=False):
    from experiments.common.spec import Workload

    shapes = ((1024, 64, 64, 64), (768, 96, 80, 48)) if holdout else ((512, 64, 64, 64), (384, 48, 80, 48))
    return [
        Workload(
            "kda_chunk_" + role, "kda_chunk_o", dict(batch=1, heads=4, sequence=s, dim=k, value_dim=v, chunk_size=c), config_space="large"
        )
        for role, (s, k, v, c) in zip(("regular", "tails"), shapes)
    ]


def training_cases():
    from experiments.common.spec import Workload

    return [
        Workload(
            "kda_chunk_" + split, "kda_chunk_o", dict(batch=1, heads=2, sequence=s, dim=k, value_dim=v, chunk_size=c), config_space="large"
        )
        for split, (s, k, v, c) in zip(("train_a", "train_b", "validation"), ((256, 32, 64, 32), (256, 64, 96, 64), (192, 48, 64, 48)))
    ]

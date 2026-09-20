"""Explicit support boundary for the active KDA intra-chunk operation."""


def carver_rank(workload, device, configs, top_k):
    raise NotImplementedError("no Carver template for token-parallel KDA intra-chunk")


__all__ = ["carver_rank"]

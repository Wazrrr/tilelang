"""Carver support boundary for token-parallel KDA intra coefficients."""


def support_reason(workload, device):
    del workload, device
    return "Carver has no template for token-parallel KDA intra coefficients"


def carver_rank(workload, device, configs, top_k):
    del workload, device, configs, top_k
    raise ValueError("Carver has no template for token-parallel KDA intra coefficients")

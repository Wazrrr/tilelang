"""Reuse Carver's MatmulTemplate with native FP8 operand and output types."""

from experiments.gemm.carver import rank_configs
from .spaces import support_reason


def carver_support_reason(workload, device):
    from experiments.common.baselines import carver_target_support_reason
    from experiments.common.spec import support_reason as kernel_support_reason

    return carver_target_support_reason(device.target) or support_reason(workload) or kernel_support_reason(workload, device)


def carver_rank(workload, device, configs, top_k):
    reason = carver_support_reason(workload, device)
    if reason:
        raise ValueError(reason)
    report = rank_configs(configs, **workload.parameters, dtype=workload.dtype, target=device.target, top_k=top_k, thread_key="threads")
    report.update(metric="carver_traffic_waves", score_units="byte-waves")
    return report

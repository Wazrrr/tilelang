"""Exhaustive pre-lowering tile analysis, independent of legacy Carver."""

from .config import ANALYSIS_VERSION, CarverConfig, CarverReject
from .analysis import Region, PropagationResult, analyze_prim_func, propagate_inputs
from .runtime import check_compiler_resources
from .cost import query_device_limits
from .ranking import rank_records
from .device_profile import profile_device, load_device_profile, anchor_latency

__all__ = [
    "ANALYSIS_VERSION",
    "CarverConfig",
    "CarverReject",
    "Region",
    "PropagationResult",
    "analyze_prim_func",
    "propagate_inputs",
    "check_compiler_resources",
    "rank_records",
    "query_device_limits",
    "profile_device",
    "load_device_profile",
    "anchor_latency",
]

"""Exhaustive pre-lowering tile analysis, independent of legacy Carver."""

from .config import ANALYSIS_VERSION, TileTuneConfig, TileTuneReject
from .src.ir import Region, PropagationResult
from .analysis import analyze_prim_func, propagate_inputs
from .runtime import check_compiler_resources
from .src.device import query_device_limits
from .ranking import rank_records
from .profiling.device_profile import profile_device, load_device_profile, anchor_latency
from .targets import TargetModel, resolve_target, current_target

__all__ = [
    "ANALYSIS_VERSION",
    "TileTuneConfig",
    "TileTuneReject",
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
    "TargetModel",
    "resolve_target",
    "current_target",
]

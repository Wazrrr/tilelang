"""Config selector registry and helpers for autotune candidate preselection."""

from .hopper_hybrid_v1 import (
    SUPPORTED_SELECTORS,
    SelectorTelemetry,
    default_selector_pool_k,
    is_hopper_or_newer_cuda_arch,
    select_configs,
    select_hints,
    should_activate_selector,
    validate_selector_name,
)

__all__ = [
    "SUPPORTED_SELECTORS",
    "SelectorTelemetry",
    "default_selector_pool_k",
    "is_hopper_or_newer_cuda_arch",
    "select_configs",
    "select_hints",
    "should_activate_selector",
    "validate_selector_name",
]

"""Deprecated compatibility aliases for the old CUDA autotune filter names."""

from tilelang.autotuner.filters import verify as _verify

for _name in _verify.__all__:
    globals()[_name] = getattr(_verify, _name)

AutotuneQualityFilterConfig = _verify.AutotuneFilterConfig
AutotuneQualityFilterReject = _verify.AutotuneFilterReject
AutotuneQualityFilterDecision = _verify.AutotuneFilterResult
CudaKernelQualityInfo = _verify.CudaKernelFilterInfo
evaluate_post_compile_quality_filter = _verify.evaluate_post_compile_filter
evaluate_tir_quality_filter = _verify.evaluate_pre_compile_filter
extract_cuda_kernel_quality_info = _verify.extract_cuda_kernel_filter_info
extract_tir_kernel_quality_info = _verify.extract_pre_compile_filter_info
iter_quality_rules = _verify.iter_filter_rules
register_quality_rule = _verify.register_filter_rule
unregister_quality_rule = _verify.unregister_filter_rule

__all__ = [
    *_verify.__all__,
    "AutotuneQualityFilterConfig",
    "AutotuneQualityFilterReject",
    "AutotuneQualityFilterDecision",
    "CudaKernelQualityInfo",
    "evaluate_post_compile_quality_filter",
    "evaluate_tir_quality_filter",
    "extract_cuda_kernel_quality_info",
    "extract_tir_kernel_quality_info",
    "iter_quality_rules",
    "register_quality_rule",
    "unregister_quality_rule",
]

"""Exact post-compile CUDA quality filtering for autotune candidates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tilelang.autotuner.filters.common import (
    AutotuneBaseFilterConfig,
    AutotuneFilterDecision,
    FilterAction,
    FilterVerdict,
    KernelType,
)
from tilelang.autotuner.filters.resources import LaunchResourceInfo

QualityFilterStage = Literal["post_compile_quality"]
QualityFilterVerdict = FilterVerdict
QualityFilterAction = FilterAction


@dataclass(frozen=True)
class AutotuneQualityFilterConfig(AutotuneBaseFilterConfig):
    """Configuration for exact post-compile CUDA quality filtering.

    The quality filter is intentionally separate from the hard resource filter:
    resource filtering rejects candidates that cannot legally launch, while this
    filter rejects or reports candidates whose emitted CUDA has exact performance
    risk signals such as spills, high accumulator footprint, or bad WGMMA shape.
    """

    enabled: bool = False
    action: FilterAction = "reject"
    report_path: str | None = None

    check_spills: bool = True
    max_spills: int | None = 0

    check_local_memory: bool = True
    max_local_size_bytes: int | None = 0

    check_registers: bool = True
    max_registers_per_thread: int | None = None

    check_c_local: bool = True
    max_c_local_floats: int | None = 256

    check_output_elements_per_thread: bool = True
    max_output_elements_per_thread: int | None = 256

    check_wgmma_n: bool = True
    max_wgmma_n: int | None = None
    check_wgmma_n_advisory: bool = True
    advisory_max_wgmma_n: int | None = 128

    check_k_loop: bool = True
    max_k_loop_iterations: int | None = None
    check_k_loop_advisory: bool = True
    advisory_max_k_loop_iterations: int | None = 64

    check_tma_tiny_tile: bool = True
    max_tma_tiny_tile_area: int | None = 4096
    tma_tiny_tile_num_stages: int | None = 1

    check_tma_store_count: bool = True
    max_tma_store_count: int | None = None

    check_quant_dequant_elements_per_thread: bool = True
    max_quant_dequant_elements_per_thread: int | None = None
    check_quant_dequant_elements_per_thread_advisory: bool = True
    advisory_max_quant_dequant_elements_per_thread: int | None = 128

    check_sparse_mask: bool = False
    check_sparse_mask_advisory: bool = True

    check_attention_spills: bool = True
    max_attention_spills: int | None = 128
    check_attention_spills_advisory: bool = True
    advisory_max_attention_spills: int | None = 0

    check_attention_local_memory: bool = True
    max_attention_local_size_bytes: int | None = 256
    check_attention_local_memory_advisory: bool = True
    advisory_max_attention_local_size_bytes: int | None = 0

    check_attention_state_elements_per_thread: bool = True
    max_attention_state_elements_per_thread: int | None = 256

    kernel_type: KernelType = "auto"

    def needs_cuda_resource_usage(self) -> bool:
        return self.enabled and (
            self.check_spills
            or self.check_local_memory
            or self.check_registers
            or self.check_attention_spills
            or self.check_attention_spills_advisory
            or self.check_attention_local_memory
            or self.check_attention_local_memory_advisory
        )


@dataclass(frozen=True)
class CudaKernelQualityInfo:
    """Exact CUDA source/PTXAS features used by the quality filter."""

    function_name: str
    source_available: bool = False
    detected_kernel_type: KernelType = "generic"
    fragment_array_elements: dict[str, int] = field(default_factory=dict)
    c_local_floats: int | None = None
    quant_dequant_elements_per_thread: int | None = None
    sparse_mask_access_count: int = 0
    attention_score_elements_per_thread: int | None = None
    attention_output_elements_per_thread: int | None = None
    attention_softmax_elements_per_thread: int | None = None
    attention_state_elements_per_thread: int | None = None
    attention_cast_elements_per_thread: int | None = None
    wgmma_shapes: list[tuple[int, int, int]] = field(default_factory=list)
    max_wgmma_n: int | None = None
    max_k_loop_iterations: int | None = None
    tma_load_count: int = 0
    tma_store_count: int = 0
    stmatrix_count: int = 0
    mbarrier_count: int = 0
    syncthreads_count: int = 0
    output_elements_per_thread: int | None = None
    tile_area: int | None = None
    num_stages: int | None = None
    n_regs: int = 0
    n_spills: int = 0
    local_size_bytes: int = 0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["wgmma_shapes"] = [list(shape) for shape in self.wgmma_shapes]
        return data


class AutotuneQualityFilterDecision(AutotuneFilterDecision):
    """Decision type for post-compile quality filtering."""

    @classmethod
    def keep_decision(cls, reason: str, **details: Any) -> "AutotuneQualityFilterDecision":
        return cls("keep", "post_compile_quality", reason, details)

    @classmethod
    def reject_decision(cls, reason: str, **details: Any) -> "AutotuneQualityFilterDecision":
        return cls("reject", "post_compile_quality", reason, details)


class AutotuneQualityFilterReject(RuntimeError):
    """Internal marker for configs skipped by exact quality filtering."""

    def __init__(
        self,
        decision: AutotuneQualityFilterDecision,
        resource_decisions: list[AutotuneFilterDecision] | None = None,
        quality_decisions: list[AutotuneQualityFilterDecision] | None = None,
    ):
        self.decision = decision
        self.resource_decisions = list(resource_decisions) if resource_decisions is not None else []
        self.quality_decisions = list(quality_decisions) if quality_decisions is not None else [decision]
        super().__init__(f"{decision.stage}:{decision.reason}:{decision.details}")


_C_LOCAL_RE = re.compile(r"\bfloat\s+\w*C(?:t)?_local(?:_\w*)?\s*\[\s*(\d+)\s*\]")
_LOCAL_ARRAY_RE = re.compile(
    r"\b(?P<type>float|half_t|half|__half|uint32_t|int|unsigned\s+int)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<count>\d+)\s*\]"
)
_CPP_WGMMA_RE = re.compile(
    r"tl::wgmma_[a-z_]*<[^;]*?kFloat32\s*,\s*(?P<m>\d+)\s*,\s*(?P<n>\d+)\s*,\s*(?P<k>\d+)",
    re.DOTALL,
)
_PTX_WGMMA_RE = re.compile(r"\bwgmma\.mma_async[^\n;]*?\.m(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)")
_K_LOOP_RE = re.compile(r"\bfor\s*\(\s*int\s+k(?:_\d+)?\s*=\s*0\s*;\s*k(?:_\d+)?\s*<\s*(\d+)\s*;")
_ATTENTION_SOFTMAX_NAMES = frozenset(
    {
        "logsum",
        "scores_max",
        "scores_max_prev",
        "scores_max_clear",
        "scores_scale",
        "scores_sum",
    }
)


def extract_cuda_function_source(kernel_source: str, function_name: str) -> str:
    """Return one CUDA kernel function body from a possibly grouped source."""
    if not kernel_source:
        return ""

    pattern = re.compile(
        r'extern\s+"C"\s+__global__\s+void\s+'
        r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
        + re.escape(function_name)
        + r"\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    match = pattern.search(kernel_source)
    if match is None:
        return kernel_source

    start = match.start()
    brace_start = kernel_source.find("{", match.end() - 1)
    if brace_start < 0:
        return kernel_source[start:]

    depth = 0
    for pos in range(brace_start, len(kernel_source)):
        char = kernel_source[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return kernel_source[start : pos + 1]
    return kernel_source[start:]


def extract_cuda_kernel_quality_info(
    function_name: str,
    kernel_source: str,
    launch_info: LaunchResourceInfo,
    raw_usage: Any,
    config: dict[str, Any] | None = None,
) -> CudaKernelQualityInfo:
    """Extract exact source and PTXAS features for one compiled CUDA kernel."""
    function_source = extract_cuda_function_source(kernel_source, function_name)
    config = config or {}

    c_local_matches = [int(match.group(1)) for match in _C_LOCAL_RE.finditer(function_source)]
    fragment_array_elements = _extract_local_array_elements(function_source)
    quant_dequant_elements = max(
        (count for name, count in fragment_array_elements.items() if _is_dequant_fragment_name(name)),
        default=None,
    )
    attention_score_elements = fragment_array_elements.get("acc_s")
    attention_output_elements = fragment_array_elements.get("acc_o")
    attention_softmax_elements = sum(fragment_array_elements.get(name, 0) for name in _ATTENTION_SOFTMAX_NAMES)
    attention_cast_elements = fragment_array_elements.get("acc_s_cast")
    attention_state_elements = (
        attention_score_elements + attention_output_elements + attention_softmax_elements
        if attention_score_elements is not None and attention_output_elements is not None
        else None
    )
    wgmma_shapes = _extract_wgmma_shapes(function_source)
    k_loop_counts = [int(match.group(1)) for match in _K_LOOP_RE.finditer(function_source)]

    block_m = _config_int(config, "block_M")
    block_n = _config_int(config, "block_N")
    thread_num = _config_int(config, "thread_num")
    if thread_num is None:
        thread_num = _config_int(config, "threads")
    if thread_num is None:
        thread_num = launch_info.threads_per_block
    tile_area = block_m * block_n if block_m is not None and block_n is not None else None
    output_elements_per_thread = (
        tile_area // thread_num
        if tile_area is not None and thread_num is not None and thread_num > 0 and tile_area % thread_num == 0
        else None
    )

    extra = _usage_extra(raw_usage)
    return CudaKernelQualityInfo(
        function_name=function_name,
        source_available=bool(function_source),
        detected_kernel_type=_detect_kernel_type(c_local_matches, fragment_array_elements, function_source),
        fragment_array_elements=fragment_array_elements,
        c_local_floats=max(c_local_matches) if c_local_matches else None,
        quant_dequant_elements_per_thread=quant_dequant_elements,
        sparse_mask_access_count=_count_sparse_mask_accesses(function_source),
        attention_score_elements_per_thread=attention_score_elements,
        attention_output_elements_per_thread=attention_output_elements,
        attention_softmax_elements_per_thread=attention_softmax_elements if attention_state_elements is not None else None,
        attention_state_elements_per_thread=attention_state_elements,
        attention_cast_elements_per_thread=attention_cast_elements,
        wgmma_shapes=wgmma_shapes,
        max_wgmma_n=max((shape[1] for shape in wgmma_shapes), default=None),
        max_k_loop_iterations=max(k_loop_counts) if k_loop_counts else None,
        tma_load_count=function_source.count("tl::tma_load("),
        tma_store_count=function_source.count("tl::tma_store("),
        stmatrix_count=function_source.count("ptx_stmatrix"),
        mbarrier_count=function_source.count("mbarrier["),
        syncthreads_count=function_source.count("__syncthreads"),
        output_elements_per_thread=output_elements_per_thread,
        tile_area=tile_area,
        num_stages=_config_int(config, "num_stages"),
        n_regs=_usage_int(raw_usage, "n_regs"),
        n_spills=_usage_int(raw_usage, "n_spills"),
        local_size_bytes=_usage_int(raw_usage, "local_size_bytes"),
        spill_stores_bytes=int(extra.get("spill_stores_bytes", 0) or 0),
        spill_loads_bytes=int(extra.get("spill_loads_bytes", 0) or 0),
    )


def evaluate_post_compile_quality_filter(
    launch_infos: list[LaunchResourceInfo],
    resource_usage: dict[str, Any],
    kernel_source: str,
    config: dict[str, Any],
    quality_config: AutotuneQualityFilterConfig,
) -> AutotuneQualityFilterDecision:
    """Evaluate exact post-compile CUDA quality targets."""
    if not quality_config.enabled:
        return AutotuneQualityFilterDecision.keep_decision("quality_filter_disabled")

    infos: list[CudaKernelQualityInfo] = []
    for launch in launch_infos:
        raw_usage = resource_usage.get(launch.function_name) if resource_usage else None
        infos.append(
            extract_cuda_kernel_quality_info(
                function_name=launch.function_name,
                kernel_source=kernel_source,
                launch_info=launch,
                raw_usage=raw_usage,
                config=config,
            )
        )

    violations = []
    advisories = []
    for info in infos:
        profile = _select_quality_profile(info, quality_config)
        violations.extend(profile.find_violations(info, quality_config))
        advisories.extend(profile.find_advisories(info, quality_config))

    details = {
        "action": quality_config.action,
        "kernels": [info.to_dict() for info in infos],
        "violations": violations,
        "advisories": advisories,
    }
    if violations and quality_config.action == "reject":
        return AutotuneQualityFilterDecision.reject_decision("quality_target_violation", **details)
    if violations:
        return AutotuneQualityFilterDecision.keep_decision("quality_report_only", **details)
    if advisories:
        return AutotuneQualityFilterDecision.keep_decision("quality_advisory_only", **details)
    return AutotuneQualityFilterDecision.keep_decision("quality_targets_passed", **details)


class _CudaQualityProfile:
    kernel_type: KernelType = "generic"

    def find_violations(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def find_advisories(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _with_kernel_type(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for finding in findings:
            finding.setdefault("kernel_type", self.kernel_type)
        return findings


class _GenericCudaQualityProfile(_CudaQualityProfile):
    kernel_type: KernelType = "generic"

    def find_violations(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        violations = _find_common_quality_violations(info, config)
        return self._with_kernel_type(violations)

    def find_advisories(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        advisories = _find_common_quality_advisories(info, config)
        return self._with_kernel_type(advisories)


class _GemmCudaQualityProfile(_GenericCudaQualityProfile):
    kernel_type: KernelType = "gemm"


class _DenseGemmCudaQualityProfile(_GemmCudaQualityProfile):
    kernel_type: KernelType = "dense_gemm"


class _QuantizedGemmCudaQualityProfile(_GemmCudaQualityProfile):
    kernel_type: KernelType = "quantized_gemm"

    def find_violations(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        violations = super().find_violations(info, config)
        _append_limit_violation(
            violations,
            enabled=config.check_quant_dequant_elements_per_thread,
            observed=info.quant_dequant_elements_per_thread,
            limit=config.max_quant_dequant_elements_per_thread,
            reason="quant_dequant_elements_per_thread_over_quality_limit",
            function=info.function_name,
        )
        return self._with_kernel_type(violations)

    def find_advisories(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        advisories = super().find_advisories(info, config)
        _append_limit_violation(
            advisories,
            enabled=config.check_quant_dequant_elements_per_thread_advisory,
            observed=info.quant_dequant_elements_per_thread,
            limit=config.advisory_max_quant_dequant_elements_per_thread,
            reason="quant_dequant_elements_per_thread_over_advisory_limit",
            function=info.function_name,
        )
        return self._with_kernel_type(advisories)


class _SparseGemmCudaQualityProfile(_GemmCudaQualityProfile):
    kernel_type: KernelType = "sparse_gemm"

    def find_violations(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        violations = super().find_violations(info, config)
        if config.check_sparse_mask and info.source_available and info.sparse_mask_access_count == 0:
            violations.append(
                {
                    "reason": "sparse_mask_not_detected",
                    "function": info.function_name,
                    "observed": info.sparse_mask_access_count,
                }
            )
        return self._with_kernel_type(violations)

    def find_advisories(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        advisories = super().find_advisories(info, config)
        if config.check_sparse_mask_advisory and info.source_available and info.sparse_mask_access_count > 0:
            advisories.append(
                {
                    "reason": "sparse_mask_guard_detected",
                    "function": info.function_name,
                    "observed": info.sparse_mask_access_count,
                }
            )
        return self._with_kernel_type(advisories)


class _AttentionCudaQualityProfile(_CudaQualityProfile):
    kernel_type: KernelType = "attention"

    def find_violations(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        _append_limit_violation(
            violations,
            enabled=config.check_attention_spills,
            observed=info.n_spills,
            limit=config.max_attention_spills,
            reason="attention_spills_over_quality_limit",
            function=info.function_name,
        )
        _append_limit_violation(
            violations,
            enabled=config.check_attention_local_memory,
            observed=info.local_size_bytes,
            limit=config.max_attention_local_size_bytes,
            reason="attention_local_memory_over_quality_limit",
            function=info.function_name,
        )
        _append_limit_violation(
            violations,
            enabled=config.check_registers,
            observed=info.n_regs,
            limit=config.max_registers_per_thread,
            reason="registers_per_thread_over_quality_limit",
            function=info.function_name,
        )
        _append_limit_violation(
            violations,
            enabled=config.check_attention_state_elements_per_thread,
            observed=info.attention_state_elements_per_thread,
            limit=config.max_attention_state_elements_per_thread,
            reason="attention_state_elements_per_thread_over_quality_limit",
            function=info.function_name,
        )
        _append_limit_violation(
            violations,
            enabled=config.check_wgmma_n,
            observed=info.max_wgmma_n,
            limit=config.max_wgmma_n,
            reason="wgmma_n_over_quality_limit",
            function=info.function_name,
        )
        _append_limit_violation(
            violations,
            enabled=config.check_k_loop,
            observed=info.max_k_loop_iterations,
            limit=config.max_k_loop_iterations,
            reason="k_loop_iterations_over_quality_limit",
            function=info.function_name,
        )
        return self._with_kernel_type(violations)

    def find_advisories(
        self,
        info: CudaKernelQualityInfo,
        config: AutotuneQualityFilterConfig,
    ) -> list[dict[str, Any]]:
        advisories: list[dict[str, Any]] = []
        _append_limit_violation(
            advisories,
            enabled=config.check_attention_spills_advisory,
            observed=info.n_spills,
            limit=config.advisory_max_attention_spills,
            reason="attention_spills_over_advisory_limit",
            function=info.function_name,
        )
        _append_limit_violation(
            advisories,
            enabled=config.check_attention_local_memory_advisory,
            observed=info.local_size_bytes,
            limit=config.advisory_max_attention_local_size_bytes,
            reason="attention_local_memory_over_advisory_limit",
            function=info.function_name,
        )
        if (
            config.check_tma_tiny_tile
            and config.max_tma_tiny_tile_area is not None
            and info.tile_area is not None
            and info.tma_load_count > 0
            and info.tile_area <= config.max_tma_tiny_tile_area
            and (config.tma_tiny_tile_num_stages is None or info.num_stages == config.tma_tiny_tile_num_stages)
        ):
            advisories.append(
                {
                    "reason": "tma_tiny_tile_advisory_limit",
                    "function": info.function_name,
                    "tile_area": info.tile_area,
                    "limit": config.max_tma_tiny_tile_area,
                    "num_stages": info.num_stages,
                    "tma_load_count": info.tma_load_count,
                }
            )
        advisories.extend(_find_common_quality_advisories(info, config))
        return self._with_kernel_type(advisories)


_QUALITY_PROFILES: dict[KernelType, _CudaQualityProfile] = {
    "generic": _GenericCudaQualityProfile(),
    "gemm": _GemmCudaQualityProfile(),
    "dense_gemm": _DenseGemmCudaQualityProfile(),
    "quantized_gemm": _QuantizedGemmCudaQualityProfile(),
    "sparse_gemm": _SparseGemmCudaQualityProfile(),
    "attention": _AttentionCudaQualityProfile(),
}


def _select_quality_profile(
    info: CudaKernelQualityInfo,
    config: AutotuneQualityFilterConfig,
) -> _CudaQualityProfile:
    kernel_type = info.detected_kernel_type if config.kernel_type == "auto" else config.kernel_type
    if kernel_type == "auto":
        kernel_type = "generic"
    return _QUALITY_PROFILES.get(kernel_type, _QUALITY_PROFILES["generic"])


def _extract_wgmma_shapes(source: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for match in _CPP_WGMMA_RE.finditer(source):
        shapes.append((int(match.group("m")), int(match.group("n")), int(match.group("k"))))
    for match in _PTX_WGMMA_RE.finditer(source):
        shapes.append((int(match.group("m")), int(match.group("n")), int(match.group("k"))))
    return shapes


def _extract_local_array_elements(source: str) -> dict[str, int]:
    arrays: dict[str, int] = {}
    for match in _LOCAL_ARRAY_RE.finditer(source):
        name = match.group("name")
        count = int(match.group("count"))
        arrays[name] = max(count, arrays.get(name, 0))
    return arrays


def _detect_kernel_type(
    c_local_matches: list[int],
    fragment_array_elements: dict[str, int],
    source: str,
) -> KernelType:
    if "acc_s" in fragment_array_elements and "acc_o" in fragment_array_elements:
        return "attention"
    if any(_is_dequant_fragment_name(name) for name in fragment_array_elements):
        return "quantized_gemm"
    if _count_sparse_mask_accesses(source) > 0:
        return "sparse_gemm"
    if c_local_matches:
        return "dense_gemm"
    return "generic"


def _is_dequant_fragment_name(name: str) -> bool:
    return "dequant" in name.lower()


def _count_sparse_mask_accesses(source: str) -> int:
    return source.count("BlockMask") + source.count("block_mask") + source.count("blockMask")


def _find_common_quality_violations(info: CudaKernelQualityInfo, config: AutotuneQualityFilterConfig) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    _append_limit_violation(
        violations,
        enabled=config.check_spills,
        observed=info.n_spills,
        limit=config.max_spills,
        reason="spills_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_local_memory,
        observed=info.local_size_bytes,
        limit=config.max_local_size_bytes,
        reason="local_memory_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_registers,
        observed=info.n_regs,
        limit=config.max_registers_per_thread,
        reason="registers_per_thread_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_c_local,
        observed=info.c_local_floats,
        limit=config.max_c_local_floats,
        reason="c_local_floats_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_output_elements_per_thread,
        observed=info.output_elements_per_thread,
        limit=config.max_output_elements_per_thread,
        reason="output_elements_per_thread_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_wgmma_n,
        observed=info.max_wgmma_n,
        limit=config.max_wgmma_n,
        reason="wgmma_n_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_k_loop,
        observed=info.max_k_loop_iterations,
        limit=config.max_k_loop_iterations,
        reason="k_loop_iterations_over_quality_limit",
        function=info.function_name,
    )
    _append_limit_violation(
        violations,
        enabled=config.check_tma_store_count,
        observed=info.tma_store_count,
        limit=config.max_tma_store_count,
        reason="tma_store_count_over_quality_limit",
        function=info.function_name,
    )

    if (
        config.check_tma_tiny_tile
        and config.max_tma_tiny_tile_area is not None
        and info.tile_area is not None
        and info.tma_load_count > 0
        and info.tile_area <= config.max_tma_tiny_tile_area
        and (config.tma_tiny_tile_num_stages is None or info.num_stages == config.tma_tiny_tile_num_stages)
    ):
        violations.append(
            {
                "reason": "tma_tiny_tile_quality_limit",
                "function": info.function_name,
                "tile_area": info.tile_area,
                "limit": config.max_tma_tiny_tile_area,
                "num_stages": info.num_stages,
                "tma_load_count": info.tma_load_count,
            }
        )

    return violations


def _find_common_quality_advisories(info: CudaKernelQualityInfo, config: AutotuneQualityFilterConfig) -> list[dict[str, Any]]:
    advisories: list[dict[str, Any]] = []
    if config.max_wgmma_n is None:
        _append_limit_violation(
            advisories,
            enabled=config.check_wgmma_n_advisory,
            observed=info.max_wgmma_n,
            limit=config.advisory_max_wgmma_n,
            reason="wgmma_n_over_advisory_limit",
            function=info.function_name,
        )
    if config.max_k_loop_iterations is None:
        _append_limit_violation(
            advisories,
            enabled=config.check_k_loop_advisory,
            observed=info.max_k_loop_iterations,
            limit=config.advisory_max_k_loop_iterations,
            reason="k_loop_iterations_over_advisory_limit",
            function=info.function_name,
        )
    return advisories


def _append_limit_violation(
    violations: list[dict[str, Any]],
    *,
    enabled: bool,
    observed: int | None,
    limit: int | None,
    reason: str,
    function: str,
) -> None:
    if not enabled or observed is None or limit is None:
        return
    if observed > limit:
        violations.append(
            {
                "reason": reason,
                "function": function,
                "observed": observed,
                "limit": limit,
            }
        )


def _config_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_int(raw_usage: Any, key: str) -> int:
    if raw_usage is None:
        return 0
    if isinstance(raw_usage, dict):
        value = raw_usage.get(key, 0)
    else:
        value = getattr(raw_usage, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _usage_extra(raw_usage: Any) -> dict[str, Any]:
    if raw_usage is None:
        return {}
    if isinstance(raw_usage, dict):
        return dict(raw_usage.get("extra", {}) or {})
    return dict(getattr(raw_usage, "extra", {}) or {})

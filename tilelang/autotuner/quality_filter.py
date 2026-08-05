"""Exact post-compile CUDA quality filtering for autotune candidates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tilelang.autotuner.resource_filter import LaunchResourceInfo

QualityFilterStage = Literal["post_compile_quality"]
QualityFilterVerdict = Literal["keep", "reject"]
QualityFilterAction = Literal["reject", "report"]


@dataclass(frozen=True)
class AutotuneQualityFilterConfig:
    """Configuration for exact post-compile CUDA quality filtering.

    The quality filter is intentionally separate from the hard resource filter:
    resource filtering rejects candidates that cannot legally launch, while this
    filter rejects or reports candidates whose emitted CUDA has exact performance
    risk signals such as spills, high accumulator footprint, or bad WGMMA shape.
    """

    enabled: bool = False
    action: QualityFilterAction = "reject"
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

    @classmethod
    def from_value(
        cls,
        value: bool | dict[str, Any] | "AutotuneQualityFilterConfig" | None,
    ) -> "AutotuneQualityFilterConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported quality filter config: {type(value)!r}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["report_path"] is not None:
            data["report_path"] = str(data["report_path"])
        return data

    def to_cache_key_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("report_path", None)
        return data

    def needs_cuda_resource_usage(self) -> bool:
        return self.enabled and (
            self.check_spills
            or self.check_local_memory
            or self.check_registers
        )


@dataclass(frozen=True)
class CudaKernelQualityInfo:
    """Exact CUDA source/PTXAS features used by the quality filter."""

    function_name: str
    source_available: bool = False
    c_local_floats: int | None = None
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


@dataclass(frozen=True)
class AutotuneQualityFilterDecision:
    verdict: QualityFilterVerdict
    stage: QualityFilterStage
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        return self.verdict == "keep"

    @classmethod
    def keep_decision(cls, reason: str, **details: Any) -> "AutotuneQualityFilterDecision":
        return cls("keep", "post_compile_quality", reason, details)

    @classmethod
    def reject_decision(cls, reason: str, **details: Any) -> "AutotuneQualityFilterDecision":
        return cls("reject", "post_compile_quality", reason, details)


class AutotuneQualityFilterReject(RuntimeError):
    """Internal marker for configs skipped by exact quality filtering."""

    def __init__(self, decision: AutotuneQualityFilterDecision):
        self.decision = decision
        super().__init__(f"{decision.stage}:{decision.reason}:{decision.details}")


_C_LOCAL_RE = re.compile(r"\bfloat\s+C_local\s*\[\s*(\d+)\s*\]")
_CPP_WGMMA_RE = re.compile(
    r"tl::wgmma_[a-z_]*<[^;]*?kFloat32\s*,\s*(?P<m>\d+)\s*,\s*(?P<n>\d+)\s*,\s*(?P<k>\d+)",
    re.DOTALL,
)
_PTX_WGMMA_RE = re.compile(r"\bwgmma\.mma_async[^\n;]*?\.m(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)")
_K_LOOP_RE = re.compile(r"\bfor\s*\(\s*int\s+k(?:_\d+)?\s*=\s*0\s*;\s*k(?:_\d+)?\s*<\s*(\d+)\s*;")


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
    wgmma_shapes = _extract_wgmma_shapes(function_source)
    k_loop_counts = [int(match.group(1)) for match in _K_LOOP_RE.finditer(function_source)]

    block_m = _config_int(config, "block_M")
    block_n = _config_int(config, "block_N")
    thread_num = _config_int(config, "thread_num")
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
        c_local_floats=max(c_local_matches) if c_local_matches else None,
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
        violations.extend(_find_quality_violations(info, quality_config))
        advisories.extend(_find_quality_advisories(info, quality_config))

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


def _extract_wgmma_shapes(source: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for match in _CPP_WGMMA_RE.finditer(source):
        shapes.append((int(match.group("m")), int(match.group("n")), int(match.group("k"))))
    for match in _PTX_WGMMA_RE.finditer(source):
        shapes.append((int(match.group("m")), int(match.group("n")), int(match.group("k"))))
    return shapes


def _find_quality_violations(info: CudaKernelQualityInfo, config: AutotuneQualityFilterConfig) -> list[dict[str, Any]]:
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


def _find_quality_advisories(info: CudaKernelQualityInfo, config: AutotuneQualityFilterConfig) -> list[dict[str, Any]]:
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

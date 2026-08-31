"""Rule-based CUDA filtering for autotune candidates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tvm.tirx.stmt_functor import post_order_visit

from tilelang.autotuner.filters.common import (
    AutotuneBaseFilterConfig,
    AutotuneFilterDecision,
    FilterAction,
    FilterVerdict,
    KernelType,
)
from tilelang.autotuner.filters.launch import LaunchResourceInfo
from tilelang.autotuner.filters.rules import AutotuneRuleContext, AutotuneRuleRegistry, AutotuneVerifyRule, RuleFindingKind, RuleLayer

FilterEvalStage = Literal["pre_compile", "post_compile"]
FilterEvalVerdict = FilterVerdict
FilterEvalAction = FilterAction

__all__ = [
    "AutotuneFilterConfig",
    "AutotuneFilterReject",
    "AutotuneFilterResult",
    "CudaKernelFilterInfo",
    "KernelClassification",
    "classify_kernel_filter_info",
    "evaluate_post_compile_filter",
    "evaluate_pre_compile_filter",
    "extract_cuda_function_source",
    "extract_cuda_kernel_filter_info",
    "extract_pre_compile_filter_info",
    "iter_filter_rules",
    "register_filter_rule",
    "unregister_filter_rule",
]


def _normalize_name_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return tuple(normalized)


@dataclass(frozen=True)
class AutotuneFilterConfig(AutotuneBaseFilterConfig):
    """Configuration for rule-based CUDA candidate filtering.

    Rules are classified as common, primitive-specific, or kernel-specific.
    The verifier rejects or reports candidates whose pre-compile IR, emitted CUDA,
    or PTXAS usage has exact performance risk signals such as spills, local memory,
    high accumulator footprint, or bad WGMMA/TMA shape choices.
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
    kernel_traits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_traits", _normalize_name_tuple(self.kernel_traits))

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
class CudaKernelFilterInfo:
    """Exact CUDA source/PTXAS features used by autotune filters."""

    function_name: str
    source_available: bool = False
    detected_kernel_type: KernelType = "generic"
    detected_kernel_traits: tuple[str, ...] = ()
    classification_evidence: tuple[str, ...] = ()
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
        data["detected_kernel_traits"] = list(self.detected_kernel_traits)
        data["classification_evidence"] = list(self.classification_evidence)
        data["wgmma_shapes"] = [list(shape) for shape in self.wgmma_shapes]
        return data


@dataclass(frozen=True)
class KernelClassification:
    """Resolved kernel type hierarchy and traits used for rule dispatch."""

    primary_kernel_type: str = "generic"
    kernel_type_tags: tuple[str, ...] = ("generic",)
    traits: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_kernel_type": self.primary_kernel_type,
            "kernel_type_tags": list(self.kernel_type_tags),
            "traits": list(self.traits),
            "evidence": list(self.evidence),
        }


class AutotuneFilterResult(AutotuneFilterDecision):
    """Decision type for rule-based autotune filtering."""

    @classmethod
    def keep_decision(
        cls,
        reason: str,
        stage: FilterEvalStage = "post_compile",
        **details: Any,
    ) -> AutotuneFilterResult:
        return cls("keep", stage, reason, details)

    @classmethod
    def reject_decision(
        cls,
        reason: str,
        stage: FilterEvalStage = "post_compile",
        **details: Any,
    ) -> AutotuneFilterResult:
        return cls("reject", stage, reason, details)


class AutotuneFilterReject(RuntimeError):
    """Internal marker for configs skipped by rule-based autotune filtering."""

    def __init__(
        self,
        decision: AutotuneFilterResult,
        resource_decisions: list[AutotuneFilterDecision] | None = None,
        filter_decisions: list[AutotuneFilterResult] | None = None,
    ):
        self.decision = decision
        self.resource_decisions = list(resource_decisions) if resource_decisions is not None else []
        self.filter_decisions = list(filter_decisions) if filter_decisions is not None else [decision]
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
_WGMMA_PREFIX_RE = re.compile(r"\bm(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)\b")
_K_LOOP_RE = re.compile(r"\bfor\s*\(\s*int\s+k(?:_\d+)?\s*=\s*0\s*;\s*k(?:_\d+)?\s*<\s*(\d+)\s*;")
_PRE_COMPILE_WGMMA_OP_NAMES = frozenset(
    {
        "tl.ptx_wgmma_ss",
        "tl.ptx_wgmma_rs",
        "tl.ptx_wgmma_sp_ss",
        "tl.ptx_wgmma_sp_rs",
    }
)
_PRE_COMPILE_MATMUL_OP_NAMES = frozenset(
    {
        "tl.tileop.gemm",
        "tl.gemm",
        "tl.ptx_mma_sm70",
        "tl.ptx_mma_block_scale",
        "tl.ptx_tcgen05_mma_ss",
        "tl.ptx_tcgen05_mma_ts",
        "tl.ptx_tcgen05_mma_blockscaled_ss",
        "tl.tvm_mfma",
        "tl.tvm_rdna_wmma",
    }
) | _PRE_COMPILE_WGMMA_OP_NAMES
_PRE_COMPILE_TMA_OP_NAMES = frozenset(
    {
        "tl.tma_load",
        "tl.tileop.tma_copy",
        "tl.tma_store_arrive",
        "tl.tma_store_wait",
        "tl.create_tma_descriptor",
    }
)
_PRE_COMPILE_REDUCTION_OP_NAMES = frozenset(
    {
        "tl.tileop.finalize_reducer",
        "tl.warp_reduce_sum",
        "tl.warp_reduce_max",
        "tl.warp_reduce_min",
    }
)
_PRE_COMPILE_ATOMIC_OP_PREFIXES = (
    "tl.atomic_",
    "tl.tileop.atomic",
)
_PRE_COMPILE_SOFTMAX_OP_NAMES = frozenset(
    {
        "tl.__exp",
        "tl.__exp10",
        "tl.__log",
        "tl.__log2",
        "tl.__log10",
    }
)
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
        r"(?:__launch_bounds__\s*\([^)]*\)\s*)?" + re.escape(function_name) + r"\s*\([^)]*\)\s*\{",
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


def extract_cuda_kernel_filter_info(
    function_name: str,
    kernel_source: str,
    launch_info: LaunchResourceInfo,
    raw_usage: Any,
    config: dict[str, Any] | None = None,
) -> CudaKernelFilterInfo:
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
    tma_load_count = function_source.count("tl::tma_load(")
    tma_store_count = function_source.count("tl::tma_store(")
    stmatrix_count = function_source.count("ptx_stmatrix")
    mbarrier_count = function_source.count("mbarrier[")
    syncthreads_count = function_source.count("__syncthreads")
    sparse_mask_access_count = _count_sparse_mask_accesses(function_source)

    traits, evidence = _detect_post_compile_kernel_traits(
        c_local_matches=c_local_matches,
        fragment_array_elements=fragment_array_elements,
        source=function_source,
        wgmma_shapes=wgmma_shapes,
        tma_load_count=tma_load_count,
        tma_store_count=tma_store_count,
        stmatrix_count=stmatrix_count,
        mbarrier_count=mbarrier_count,
        syncthreads_count=syncthreads_count,
        sparse_mask_access_count=sparse_mask_access_count,
        config=config,
    )
    detected_kernel_type = _detect_kernel_type(
        c_local_matches,
        fragment_array_elements,
        function_source,
        traits=traits,
    )

    config_metrics = _extract_config_filter_metrics(config, launch_info)

    extra = _usage_extra(raw_usage)
    return CudaKernelFilterInfo(
        function_name=function_name,
        source_available=bool(function_source),
        detected_kernel_type=detected_kernel_type,
        detected_kernel_traits=traits,
        classification_evidence=evidence,
        fragment_array_elements=fragment_array_elements,
        c_local_floats=max(c_local_matches) if c_local_matches else None,
        quant_dequant_elements_per_thread=quant_dequant_elements,
        sparse_mask_access_count=sparse_mask_access_count,
        attention_score_elements_per_thread=attention_score_elements,
        attention_output_elements_per_thread=attention_output_elements,
        attention_softmax_elements_per_thread=attention_softmax_elements if attention_state_elements is not None else None,
        attention_state_elements_per_thread=attention_state_elements,
        attention_cast_elements_per_thread=attention_cast_elements,
        wgmma_shapes=wgmma_shapes,
        max_wgmma_n=max((shape[1] for shape in wgmma_shapes), default=None),
        max_k_loop_iterations=max(k_loop_counts) if k_loop_counts else config_metrics["max_k_loop_iterations"],
        tma_load_count=tma_load_count,
        tma_store_count=tma_store_count,
        stmatrix_count=stmatrix_count,
        mbarrier_count=mbarrier_count,
        syncthreads_count=syncthreads_count,
        output_elements_per_thread=config_metrics["output_elements_per_thread"],
        tile_area=config_metrics["tile_area"],
        num_stages=config_metrics["num_stages"],
        n_regs=_usage_int(raw_usage, "n_regs"),
        n_spills=_usage_int(raw_usage, "n_spills"),
        local_size_bytes=_usage_int(raw_usage, "local_size_bytes"),
        spill_stores_bytes=int(extra.get("spill_stores_bytes", 0) or 0),
        spill_loads_bytes=int(extra.get("spill_loads_bytes", 0) or 0),
    )


def extract_pre_compile_filter_info(
    function_name: str,
    device_mod: Any,
    launch_info: LaunchResourceInfo,
    config: dict[str, Any] | None = None,
    kernel_source: str | None = None,
) -> CudaKernelFilterInfo:
    """Extract filter features visible before the real CUDA device compile.

    The first half comes from lowered device IR.  When ``kernel_source`` is
    available, it is the CUDA emitted with ``compile_device=False`` and is still
    a pre-NVCC signal.
    """
    config = config or {}
    wgmma_shapes = _extract_pre_compile_wgmma_shapes(device_mod, function_name)
    op_names = _extract_pre_compile_op_names(device_mod, function_name)
    traits, evidence = _detect_pre_compile_kernel_traits(
        wgmma_shapes=wgmma_shapes,
        op_names=op_names,
        config=config,
    )
    detected_kernel_type = _detect_pre_compile_kernel_type(traits)
    config_metrics = _extract_config_filter_metrics(config, launch_info)
    info = CudaKernelFilterInfo(
        function_name=function_name,
        source_available=False,
        detected_kernel_type=detected_kernel_type,
        detected_kernel_traits=traits,
        classification_evidence=evidence,
        wgmma_shapes=wgmma_shapes,
        max_wgmma_n=max((shape[1] for shape in wgmma_shapes), default=None),
        max_k_loop_iterations=config_metrics["max_k_loop_iterations"],
        output_elements_per_thread=config_metrics["output_elements_per_thread"],
        tile_area=config_metrics["tile_area"],
        num_stages=config_metrics["num_stages"],
    )
    if not kernel_source:
        return info

    source_info = extract_cuda_kernel_filter_info(
        function_name=function_name,
        kernel_source=kernel_source,
        launch_info=launch_info,
        raw_usage=None,
        config=config,
    )
    return _combine_pre_compile_filter_info(info, source_info)


def evaluate_pre_compile_filter(
    launch_infos: list[LaunchResourceInfo],
    device_mod: Any,
    config: dict[str, Any],
    filter_config: AutotuneFilterConfig,
    kernel_source: str | None = None,
) -> AutotuneFilterResult:
    """Evaluate filter targets that are exact before the real CUDA compile."""
    if not filter_config.enabled:
        return AutotuneFilterResult.keep_decision("filter_disabled", stage="pre_compile")

    infos = [
        extract_pre_compile_filter_info(
            function_name=launch.function_name,
            device_mod=device_mod,
            launch_info=launch,
            config=config,
            kernel_source=kernel_source,
        )
        for launch in launch_infos
    ]

    return _evaluate_filter_infos(infos, filter_config, stage="pre_compile")


def evaluate_post_compile_filter(
    launch_infos: list[LaunchResourceInfo],
    resource_usage: dict[str, Any],
    kernel_source: str,
    config: dict[str, Any],
    filter_config: AutotuneFilterConfig,
) -> AutotuneFilterResult:
    """Evaluate exact post-compile CUDA filter targets."""
    if not filter_config.enabled:
        return AutotuneFilterResult.keep_decision("filter_disabled")

    infos: list[CudaKernelFilterInfo] = []
    for launch in launch_infos:
        raw_usage = resource_usage.get(launch.function_name) if resource_usage else None
        infos.append(
            extract_cuda_kernel_filter_info(
                function_name=launch.function_name,
                kernel_source=kernel_source,
                launch_info=launch,
                raw_usage=raw_usage,
                config=config,
            )
        )

    return _evaluate_filter_infos(infos, filter_config, stage="post_compile")


def _combine_pre_compile_filter_info(
    ir_info: CudaKernelFilterInfo,
    source_info: CudaKernelFilterInfo,
) -> CudaKernelFilterInfo:
    source_type = source_info.detected_kernel_type
    detected_kernel_type = (
        source_type if source_type not in ("auto", "generic") else ir_info.detected_kernel_type
    )
    wgmma_shapes = _merge_wgmma_shapes(ir_info.wgmma_shapes, source_info.wgmma_shapes)
    return CudaKernelFilterInfo(
        function_name=ir_info.function_name,
        source_available=source_info.source_available,
        detected_kernel_type=detected_kernel_type,
        detected_kernel_traits=_merge_names(ir_info.detected_kernel_traits, source_info.detected_kernel_traits),
        classification_evidence=_merge_names(ir_info.classification_evidence, source_info.classification_evidence),
        fragment_array_elements=source_info.fragment_array_elements,
        c_local_floats=source_info.c_local_floats,
        quant_dequant_elements_per_thread=source_info.quant_dequant_elements_per_thread,
        sparse_mask_access_count=source_info.sparse_mask_access_count,
        attention_score_elements_per_thread=source_info.attention_score_elements_per_thread,
        attention_output_elements_per_thread=source_info.attention_output_elements_per_thread,
        attention_softmax_elements_per_thread=source_info.attention_softmax_elements_per_thread,
        attention_state_elements_per_thread=source_info.attention_state_elements_per_thread,
        attention_cast_elements_per_thread=source_info.attention_cast_elements_per_thread,
        wgmma_shapes=wgmma_shapes,
        max_wgmma_n=max((shape[1] for shape in wgmma_shapes), default=None),
        max_k_loop_iterations=(
            source_info.max_k_loop_iterations
            if source_info.max_k_loop_iterations is not None
            else ir_info.max_k_loop_iterations
        ),
        tma_load_count=source_info.tma_load_count,
        tma_store_count=source_info.tma_store_count,
        stmatrix_count=source_info.stmatrix_count,
        mbarrier_count=source_info.mbarrier_count,
        syncthreads_count=source_info.syncthreads_count,
        output_elements_per_thread=(
            ir_info.output_elements_per_thread
            if ir_info.output_elements_per_thread is not None
            else source_info.output_elements_per_thread
        ),
        tile_area=ir_info.tile_area if ir_info.tile_area is not None else source_info.tile_area,
        num_stages=ir_info.num_stages if ir_info.num_stages is not None else source_info.num_stages,
    )


def _merge_wgmma_shapes(*groups: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    merged: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for group in groups:
        for shape in group:
            if shape in seen:
                continue
            merged.append(shape)
            seen.add(shape)
    return merged


def _evaluate_filter_infos(
    infos: list[CudaKernelFilterInfo],
    filter_config: AutotuneFilterConfig,
    stage: FilterEvalStage,
) -> AutotuneFilterResult:
    violations = []
    advisories = []
    classifications = []
    for info in infos:
        classification = classify_kernel_filter_info(info, filter_config)
        classifications.append(classification)
        context = AutotuneRuleContext(
            info=info,
            config=filter_config,
            stage=stage,
            primary_kernel_type=classification.primary_kernel_type,
            kernel_type_tags=frozenset(classification.kernel_type_tags),
            kernel_traits=frozenset(classification.traits),
            classification_evidence=classification.evidence,
        )
        violations.extend(_run_filter_rules(context, finding_kind="violation"))
        advisories.extend(_run_filter_rules(context, finding_kind="advisory"))

    details = {
        "action": filter_config.action,
        "kernels": [info.to_dict() for info in infos],
        "classifications": [classification.to_dict() for classification in classifications],
        "violations": violations,
        "advisories": advisories,
    }
    if violations and filter_config.action == "reject":
        return AutotuneFilterResult.reject_decision("filter_target_violation", stage=stage, **details)
    if violations:
        return AutotuneFilterResult.keep_decision("filter_report_only", stage=stage, **details)
    if advisories:
        return AutotuneFilterResult.keep_decision("filter_advisory_only", stage=stage, **details)
    return AutotuneFilterResult.keep_decision("filter_targets_passed", stage=stage, **details)


_ATTENTION_KERNEL_TYPE_TAGS = frozenset({"attention"})
_QUANTIZED_GEMM_KERNEL_TYPE_TAGS = frozenset({"quantized_gemm"})
_SPARSE_GEMM_KERNEL_TYPE_TAGS = frozenset({"sparse_gemm"})
_PRE_COMPILE_STAGES = frozenset({"pre_compile"})
_POST_COMPILE_STAGES = frozenset({"post_compile"})


class _LimitFilterRule(AutotuneVerifyRule):
    def __init__(
        self,
        *,
        name: str,
        layer: RuleLayer,
        finding_kind: RuleFindingKind,
        info_attr: str,
        enabled_attr: str,
        limit_attr: str,
        reason: str,
        match_kernel_type_tags: frozenset[str] | None = None,
        exclude_kernel_type_tags: frozenset[str] = frozenset(),
        skip_when_config_attr_set: str | None = None,
        stages: frozenset[str] | None = None,
    ):
        self.name = name
        self.layer = layer
        self.finding_kind = finding_kind
        self.info_attr = info_attr
        self.enabled_attr = enabled_attr
        self.limit_attr = limit_attr
        self.reason = reason
        self.match_kernel_type_tags = match_kernel_type_tags
        self.exclude_kernel_type_tags = exclude_kernel_type_tags
        self.skip_when_config_attr_set = skip_when_config_attr_set
        self.stages = stages

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        if self.skip_when_config_attr_set is not None and getattr(context.config, self.skip_when_config_attr_set) is not None:
            return []
        findings: list[dict[str, Any]] = []
        _append_limit_violation(
            findings,
            enabled=bool(getattr(context.config, self.enabled_attr)),
            observed=getattr(context.info, self.info_attr),
            limit=getattr(context.config, self.limit_attr),
            reason=self.reason,
            function=context.info.function_name,
        )
        return findings


class _TmaTinyTileRule(AutotuneVerifyRule):
    layer: RuleLayer = "primitive"

    def __init__(
        self,
        *,
        name: str,
        finding_kind: RuleFindingKind,
        reason: str,
        match_kernel_type_tags: frozenset[str] | None = None,
        exclude_kernel_type_tags: frozenset[str] = frozenset(),
        stages: frozenset[str] | None = None,
    ):
        self.name = name
        self.finding_kind = finding_kind
        self.reason = reason
        self.match_kernel_type_tags = match_kernel_type_tags
        self.exclude_kernel_type_tags = exclude_kernel_type_tags
        self.stages = stages

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        config = context.config
        info = context.info
        if not (
            config.check_tma_tiny_tile
            and config.max_tma_tiny_tile_area is not None
            and info.tile_area is not None
            and info.tma_load_count > 0
            and info.tile_area <= config.max_tma_tiny_tile_area
            and (config.tma_tiny_tile_num_stages is None or info.num_stages == config.tma_tiny_tile_num_stages)
        ):
            return []
        return [
            {
                "reason": self.reason,
                "function": info.function_name,
                "tile_area": info.tile_area,
                "limit": config.max_tma_tiny_tile_area,
                "num_stages": info.num_stages,
                "tma_load_count": info.tma_load_count,
            }
        ]


class _SparseMaskRequiredRule(AutotuneVerifyRule):
    name = "sparse_gemm.sparse_mask_required"
    layer: RuleLayer = "kernel"
    finding_kind: RuleFindingKind = "violation"
    match_kernel_type_tags = _SPARSE_GEMM_KERNEL_TYPE_TAGS
    stages = _PRE_COMPILE_STAGES

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        config = context.config
        info = context.info
        if config.check_sparse_mask and info.source_available and info.sparse_mask_access_count == 0:
            return [
                {
                    "reason": "sparse_mask_not_detected",
                    "function": info.function_name,
                    "observed": info.sparse_mask_access_count,
                }
            ]
        return []


class _SparseMaskAdvisoryRule(AutotuneVerifyRule):
    name = "sparse_gemm.sparse_mask_guard_advisory"
    layer: RuleLayer = "kernel"
    finding_kind: RuleFindingKind = "advisory"
    match_kernel_type_tags = _SPARSE_GEMM_KERNEL_TYPE_TAGS
    stages = _PRE_COMPILE_STAGES

    def check(self, context: AutotuneRuleContext) -> list[dict[str, Any]]:
        config = context.config
        info = context.info
        if config.check_sparse_mask_advisory and info.source_available and info.sparse_mask_access_count > 0:
            return [
                {
                    "reason": "sparse_mask_guard_detected",
                    "function": info.function_name,
                    "observed": info.sparse_mask_access_count,
                }
            ]
        return []


def _make_default_filter_rule_registry() -> AutotuneRuleRegistry:
    registry = AutotuneRuleRegistry()
    for rule in (
        _LimitFilterRule(
            name="attention.spills",
            layer="kernel",
            finding_kind="violation",
            info_attr="n_spills",
            enabled_attr="check_attention_spills",
            limit_attr="max_attention_spills",
            reason="attention_spills_over_limit",
            match_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_POST_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="attention.local_memory",
            layer="kernel",
            finding_kind="violation",
            info_attr="local_size_bytes",
            enabled_attr="check_attention_local_memory",
            limit_attr="max_attention_local_size_bytes",
            reason="attention_local_memory_over_limit",
            match_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_POST_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="common.spills",
            layer="common",
            finding_kind="violation",
            info_attr="n_spills",
            enabled_attr="check_spills",
            limit_attr="max_spills",
            reason="spills_over_limit",
            exclude_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_POST_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="common.local_memory",
            layer="common",
            finding_kind="violation",
            info_attr="local_size_bytes",
            enabled_attr="check_local_memory",
            limit_attr="max_local_size_bytes",
            reason="local_memory_over_limit",
            exclude_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_POST_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="common.registers",
            layer="common",
            finding_kind="violation",
            info_attr="n_regs",
            enabled_attr="check_registers",
            limit_attr="max_registers_per_thread",
            reason="registers_per_thread_over_limit",
            stages=_POST_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="attention.state_elements",
            layer="kernel",
            finding_kind="violation",
            info_attr="attention_state_elements_per_thread",
            enabled_attr="check_attention_state_elements_per_thread",
            limit_attr="max_attention_state_elements_per_thread",
            reason="attention_state_elements_per_thread_over_limit",
            match_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="gemm.c_local",
            layer="kernel",
            finding_kind="violation",
            info_attr="c_local_floats",
            enabled_attr="check_c_local",
            limit_attr="max_c_local_floats",
            reason="c_local_floats_over_limit",
            exclude_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="primitive.output_elements_per_thread",
            layer="primitive",
            finding_kind="violation",
            info_attr="output_elements_per_thread",
            enabled_attr="check_output_elements_per_thread",
            limit_attr="max_output_elements_per_thread",
            reason="output_elements_per_thread_over_limit",
            exclude_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="primitive.wgmma_n",
            layer="primitive",
            finding_kind="violation",
            info_attr="max_wgmma_n",
            enabled_attr="check_wgmma_n",
            limit_attr="max_wgmma_n",
            reason="wgmma_n_over_limit",
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="primitive.k_loop",
            layer="primitive",
            finding_kind="violation",
            info_attr="max_k_loop_iterations",
            enabled_attr="check_k_loop",
            limit_attr="max_k_loop_iterations",
            reason="k_loop_iterations_over_limit",
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="primitive.tma_store_count",
            layer="primitive",
            finding_kind="violation",
            info_attr="tma_store_count",
            enabled_attr="check_tma_store_count",
            limit_attr="max_tma_store_count",
            reason="tma_store_count_over_limit",
            exclude_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _TmaTinyTileRule(
            name="primitive.tma_tiny_tile",
            finding_kind="violation",
            reason="tma_tiny_tile_limit",
            exclude_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="quantized_gemm.dequant_elements",
            layer="kernel",
            finding_kind="violation",
            info_attr="quant_dequant_elements_per_thread",
            enabled_attr="check_quant_dequant_elements_per_thread",
            limit_attr="max_quant_dequant_elements_per_thread",
            reason="quant_dequant_elements_per_thread_over_limit",
            match_kernel_type_tags=_QUANTIZED_GEMM_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _SparseMaskRequiredRule(),
        _LimitFilterRule(
            name="attention.spills_advisory",
            layer="kernel",
            finding_kind="advisory",
            info_attr="n_spills",
            enabled_attr="check_attention_spills_advisory",
            limit_attr="advisory_max_attention_spills",
            reason="attention_spills_over_advisory_limit",
            match_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_POST_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="attention.local_memory_advisory",
            layer="kernel",
            finding_kind="advisory",
            info_attr="local_size_bytes",
            enabled_attr="check_attention_local_memory_advisory",
            limit_attr="advisory_max_attention_local_size_bytes",
            reason="attention_local_memory_over_advisory_limit",
            match_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_POST_COMPILE_STAGES,
        ),
        _TmaTinyTileRule(
            name="attention.tma_tiny_tile_advisory",
            finding_kind="advisory",
            reason="tma_tiny_tile_advisory_limit",
            match_kernel_type_tags=_ATTENTION_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="primitive.wgmma_n_advisory",
            layer="primitive",
            finding_kind="advisory",
            info_attr="max_wgmma_n",
            enabled_attr="check_wgmma_n_advisory",
            limit_attr="advisory_max_wgmma_n",
            reason="wgmma_n_over_advisory_limit",
            skip_when_config_attr_set="max_wgmma_n",
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="primitive.k_loop_advisory",
            layer="primitive",
            finding_kind="advisory",
            info_attr="max_k_loop_iterations",
            enabled_attr="check_k_loop_advisory",
            limit_attr="advisory_max_k_loop_iterations",
            reason="k_loop_iterations_over_advisory_limit",
            skip_when_config_attr_set="max_k_loop_iterations",
            stages=_PRE_COMPILE_STAGES,
        ),
        _LimitFilterRule(
            name="quantized_gemm.dequant_elements_advisory",
            layer="kernel",
            finding_kind="advisory",
            info_attr="quant_dequant_elements_per_thread",
            enabled_attr="check_quant_dequant_elements_per_thread_advisory",
            limit_attr="advisory_max_quant_dequant_elements_per_thread",
            reason="quant_dequant_elements_per_thread_over_advisory_limit",
            match_kernel_type_tags=_QUANTIZED_GEMM_KERNEL_TYPE_TAGS,
            stages=_PRE_COMPILE_STAGES,
        ),
        _SparseMaskAdvisoryRule(),
    ):
        registry.register(rule)
    return registry


_FILTER_RULE_REGISTRY = _make_default_filter_rule_registry()


def register_filter_rule(rule: AutotuneVerifyRule, *, replace: bool = False) -> AutotuneVerifyRule:
    """Register a custom autotune filter rule."""
    return _FILTER_RULE_REGISTRY.register(rule, replace=replace)


def unregister_filter_rule(name: str) -> AutotuneVerifyRule | None:
    """Remove a previously registered autotune filter rule."""
    return _FILTER_RULE_REGISTRY.unregister(name)


def iter_filter_rules(
    *,
    layer: RuleLayer | None = None,
    finding_kind: RuleFindingKind | None = None,
) -> list[AutotuneVerifyRule]:
    """Return the registered autotune filter rules in evaluation order."""
    return _FILTER_RULE_REGISTRY.rules(layer=layer, finding_kind=finding_kind)


def _run_filter_rules(
    context: AutotuneRuleContext,
    *,
    finding_kind: RuleFindingKind,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rule in _FILTER_RULE_REGISTRY.rules_for(context, finding_kind=finding_kind):
        for finding in rule.check(context):
            finding.setdefault("kernel_type", context.primary_kernel_type)
            findings.append(finding)
    return findings


def classify_kernel_filter_info(
    info: CudaKernelFilterInfo,
    config: AutotuneFilterConfig,
) -> KernelClassification:
    """Resolve user-provided kernel hints against automatic classifier output."""
    detected_type = str(info.detected_kernel_type or "generic")
    detected_traits = tuple(info.detected_kernel_traits)
    explicit_traits = _normalize_name_tuple(config.kernel_traits)
    evidence = list(info.classification_evidence)

    if config.kernel_type != "auto":
        primary_kernel_type = str(config.kernel_type)
        evidence.append(f"user.kernel_type={primary_kernel_type}")
    else:
        primary_kernel_type = "generic" if detected_type == "auto" else detected_type

    traits = _merge_names(detected_traits, explicit_traits)
    if explicit_traits:
        evidence.append(f"user.kernel_traits={','.join(explicit_traits)}")
    kernel_type_tags = _expand_kernel_type_tags(primary_kernel_type, traits)
    return KernelClassification(
        primary_kernel_type=primary_kernel_type,
        kernel_type_tags=kernel_type_tags,
        traits=traits,
        evidence=tuple(evidence),
    )


def _merge_names(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            if name and name not in seen:
                merged.append(name)
                seen.add(name)
    return tuple(merged)


def _expand_kernel_type_tags(
    kernel_type: str,
    traits: tuple[str, ...] = (),
) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "generic": ("generic",),
        "matmul": ("matmul",),
        "gemm": ("matmul", "gemm"),
        "dense_gemm": ("matmul", "gemm", "dense_gemm"),
        "grouped_gemm": ("matmul", "gemm", "grouped_gemm"),
        "splitk_gemm": ("matmul", "gemm", "splitk_gemm"),
        "streamk_gemm": ("matmul", "gemm", "streamk_gemm"),
        "quantized_gemm": ("matmul", "gemm", "quantized_gemm"),
        "blockscaled_gemm": ("matmul", "gemm", "quantized_gemm", "blockscaled_gemm"),
        "sparse_gemm": ("matmul", "gemm", "sparse_gemm"),
        "gemv": ("matmul", "gemv"),
        "attention": ("attention",),
        "flash_attention": ("attention", "flash_attention"),
        "flash_decoding": ("attention", "flash_decoding"),
        "block_sparse_attention": ("attention", "block_sparse_attention"),
        "mla": ("attention", "mla"),
        "nsa": ("attention", "nsa"),
        "attention_sink": ("attention", "attention_sink"),
        "reduction": ("reduction",),
        "softmax": ("reduction", "softmax"),
        "norm": ("reduction", "norm"),
        "topk": ("reduction", "topk"),
        "scan": ("scan",),
        "conv": ("conv",),
        "elementwise": ("elementwise",),
        "cast": ("elementwise", "cast"),
    }
    expanded_types = mapping.get(kernel_type)
    if expanded_types is not None:
        return expanded_types
    if "uses_gemm" in traits:
        return ("matmul", kernel_type)
    return (kernel_type,)


def _detect_post_compile_kernel_traits(
    *,
    c_local_matches: list[int],
    fragment_array_elements: dict[str, int],
    source: str,
    wgmma_shapes: list[tuple[int, int, int]],
    tma_load_count: int,
    tma_store_count: int,
    stmatrix_count: int,
    mbarrier_count: int,
    syncthreads_count: int,
    sparse_mask_access_count: int,
    config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    traits: list[str] = []
    evidence: list[str] = []

    def add(trait: str, reason: str) -> None:
        if trait not in traits:
            traits.append(trait)
        evidence.append(reason)

    if wgmma_shapes:
        add("uses_wgmma", f"source.wgmma_shapes={len(wgmma_shapes)}")
        add("uses_gemm", "source.uses_wgmma")
    if c_local_matches:
        add("uses_gemm", f"source.c_local_arrays={len(c_local_matches)}")
        add("has_accumulator_tile", f"source.max_c_local={max(c_local_matches)}")
    if any(_is_dequant_fragment_name(name) for name in fragment_array_elements) or "dequant" in source.lower():
        add("has_dequant", "source.dequant_signal")
    if sparse_mask_access_count:
        add("has_sparse_mask", f"source.sparse_mask_accesses={sparse_mask_access_count}")
    if "acc_s" in fragment_array_elements and "acc_o" in fragment_array_elements:
        add("uses_gemm", "source.attention_accumulators")
        add("has_attention_state", "source.acc_s_and_acc_o")
        add("has_softmax", "source.attention_state")
    if any(name in fragment_array_elements for name in _ATTENTION_SOFTMAX_NAMES):
        add("has_online_softmax", "source.softmax_state_fragments")
    if "exp2(" in source or "expf(" in source or "exp(" in source:
        add("has_exp", "source.exp")
    if "reduce_max" in source or "reduce_sum" in source:
        add("uses_reduction", "source.reduction")
    if tma_load_count or tma_store_count:
        add("uses_tma", f"source.tma_loads={tma_load_count},stores={tma_store_count}")
    if stmatrix_count:
        add("uses_stmatrix", f"source.stmatrix={stmatrix_count}")
    if mbarrier_count:
        add("uses_mbarrier", f"source.mbarrier={mbarrier_count}")
    if syncthreads_count:
        add("uses_syncthreads", f"source.syncthreads={syncthreads_count}")
    if _has_tiled_mnk_config(config):
        add("has_tiled_mnk_config", "config.block_M/block_N/block_K")

    return tuple(traits), tuple(evidence)


def _detect_pre_compile_kernel_traits(
    *,
    wgmma_shapes: list[tuple[int, int, int]],
    op_names: frozenset[str],
    config: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    traits: list[str] = []
    evidence: list[str] = []

    def add(trait: str, reason: str) -> None:
        if trait not in traits:
            traits.append(trait)
        evidence.append(reason)

    if wgmma_shapes:
        add("uses_wgmma", f"ir.wgmma_shapes={len(wgmma_shapes)}")
        add("uses_gemm", "ir.uses_wgmma")
    if op_names & _PRE_COMPILE_MATMUL_OP_NAMES:
        add("uses_gemm", "ir.matmul_ops")
    if op_names & _PRE_COMPILE_TMA_OP_NAMES:
        add("uses_tma", "ir.tma_ops")
    if op_names & _PRE_COMPILE_REDUCTION_OP_NAMES:
        add("uses_reduction", "ir.reduction_ops")
    if op_names & _PRE_COMPILE_SOFTMAX_OP_NAMES:
        add("has_exp_or_log", "ir.exp_or_log_ops")
    if any(op_name.startswith(_PRE_COMPILE_ATOMIC_OP_PREFIXES) for op_name in op_names):
        add("uses_atomic", "ir.atomic_ops")
    if _has_tiled_mnk_config(config):
        add("has_tiled_mnk_config", "config.block_M/block_N/block_K")

    return tuple(traits), tuple(evidence)


def _detect_pre_compile_kernel_type(traits: tuple[str, ...]) -> KernelType:
    if "uses_gemm" in traits:
        return "generic"
    if "uses_reduction" in traits and "has_exp_or_log" in traits:
        return "softmax"
    if "uses_reduction" in traits:
        return "reduction"
    return "generic"


def _extract_wgmma_shapes(source: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for match in _CPP_WGMMA_RE.finditer(source):
        shapes.append((int(match.group("m")), int(match.group("n")), int(match.group("k"))))
    for match in _PTX_WGMMA_RE.finditer(source):
        shapes.append((int(match.group("m")), int(match.group("n")), int(match.group("k"))))
    return shapes


def _extract_pre_compile_wgmma_shapes(device_mod: Any, function_name: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []

    def visit(node: Any) -> None:
        op = getattr(node, "op", None)
        op_name = getattr(op, "name", None)
        if op_name not in _PRE_COMPILE_WGMMA_OP_NAMES:
            return
        args = getattr(node, "args", None)
        if not args:
            return
        shape = _parse_wgmma_shape_text(_string_imm_value(args[0]))
        if shape is not None:
            shapes.append(shape)

    for global_var, func in getattr(device_mod, "functions", {}).items():
        attrs = getattr(func, "attrs", None) or {}
        symbol = str(attrs.get("global_symbol", getattr(global_var, "name_hint", str(global_var))))
        if symbol != function_name:
            continue
        body = getattr(func, "body", None)
        if body is not None:
            post_order_visit(body, visit)
    return shapes


def _extract_pre_compile_op_names(device_mod: Any, function_name: str) -> frozenset[str]:
    op_names: set[str] = set()

    def visit(node: Any) -> None:
        op = getattr(node, "op", None)
        op_name = getattr(op, "name", None)
        if op_name:
            op_names.add(str(op_name))

    for global_var, func in getattr(device_mod, "functions", {}).items():
        attrs = getattr(func, "attrs", None) or {}
        symbol = str(attrs.get("global_symbol", getattr(global_var, "name_hint", str(global_var))))
        if symbol != function_name:
            continue
        body = getattr(func, "body", None)
        if body is not None:
            post_order_visit(body, visit)
    return frozenset(op_names)


def _parse_wgmma_shape_text(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    match = _WGMMA_PREFIX_RE.search(text)
    if match is None:
        return None
    return (int(match.group("m")), int(match.group("n")), int(match.group("k")))


def _string_imm_value(value: Any) -> str | None:
    raw = getattr(value, "value", None)
    if raw is not None:
        return str(raw)
    if isinstance(value, str):
        return value
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _extract_config_filter_metrics(
    config: dict[str, Any],
    launch_info: LaunchResourceInfo,
) -> dict[str, int | None]:
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
    max_k_loop_iterations = _config_k_loop_iterations(config)
    return {
        "tile_area": tile_area,
        "output_elements_per_thread": output_elements_per_thread,
        "max_k_loop_iterations": max_k_loop_iterations,
        "num_stages": _config_int(config, "num_stages"),
    }


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
    *,
    traits: tuple[str, ...] = (),
) -> KernelType:
    if "acc_s" in fragment_array_elements and "acc_o" in fragment_array_elements:
        return "attention"
    if any(_is_dequant_fragment_name(name) for name in fragment_array_elements):
        return "quantized_gemm" if "uses_gemm" in traits else "generic"
    if _count_sparse_mask_accesses(source) > 0:
        return "sparse_gemm" if "uses_gemm" in traits else "generic"
    if c_local_matches:
        return "dense_gemm"
    return "generic"


def _is_dequant_fragment_name(name: str) -> bool:
    return "dequant" in name.lower()


def _count_sparse_mask_accesses(source: str) -> int:
    return source.count("BlockMask") + source.count("block_mask") + source.count("blockMask")


def _has_tiled_mnk_config(config: dict[str, Any]) -> bool:
    return (
        _config_int(config, "block_M") is not None
        and _config_int(config, "block_N") is not None
        and _config_int(config, "block_K") is not None
    )


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


def _config_k_loop_iterations(config: dict[str, Any]) -> int | None:
    k_extent = _config_int(config, "K")
    if k_extent is None:
        k_extent = _config_int(config, "k")
    block_k = _config_int(config, "block_K")
    if block_k is None:
        block_k = _config_int(config, "BLOCK_K")
    if k_extent is None or block_k is None or block_k <= 0:
        return None
    return (k_extent + block_k - 1) // block_k


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

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
from tilelang.autotuner.filters.classifier import (
    ATTENTION_SOFTMAX_NAMES,
    PRE_COMPILE_WGMMA_OP_NAMES,
    KernelClassification,
    classify_kernel_filter_info,
    detect_cuda_kernel_traits,
    detect_cuda_kernel_type,
    detect_pre_compile_kernel_traits,
    detect_pre_compile_kernel_type,
    is_dequant_fragment_name,
    merge_kernel_names,
    normalize_kernel_names,
)
from tilelang.autotuner.filters.launch import LaunchResourceInfo
from tilelang.autotuner.filters.rule_sets import make_default_filter_rules
from tilelang.autotuner.filters.rules import (
    AutotuneRuleContext,
    AutotuneRuleRegistry,
    AutotuneVerifyRule,
    RuleFindingKind,
    RuleLayer,
)

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
    max_wgmma_n: int | None = 128

    check_k_loop: bool = True
    max_k_loop_iterations: int | None = 64

    check_tma_tiny_tile: bool = True
    max_tma_tiny_tile_area: int | None = 4096
    tma_tiny_tile_num_stages: int | None = 1

    check_tma_store_count: bool = True
    max_tma_store_count: int | None = None

    check_quant_dequant_elements_per_thread: bool = True
    max_quant_dequant_elements_per_thread: int | None = 128

    check_sparse_mask: bool = False

    check_attention_spills: bool = True
    max_attention_spills: int | None = 128

    check_attention_local_memory: bool = True
    max_attention_local_size_bytes: int | None = 256

    check_attention_state_elements_per_thread: bool = True
    max_attention_state_elements_per_thread: int | None = 256

    kernel_type: KernelType = "auto"
    kernel_traits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_traits", normalize_kernel_names(self.kernel_traits))

    def needs_cuda_resource_usage(self) -> bool:
        return self.enabled and (
            self.check_spills
            or self.check_local_memory
            or self.check_registers
            or self.check_attention_spills
            or self.check_attention_local_memory
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
        (count for name, count in fragment_array_elements.items() if is_dequant_fragment_name(name)),
        default=None,
    )
    attention_score_elements = fragment_array_elements.get("acc_s")
    attention_output_elements = fragment_array_elements.get("acc_o")
    attention_softmax_elements = sum(fragment_array_elements.get(name, 0) for name in ATTENTION_SOFTMAX_NAMES)
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

    traits, evidence = detect_cuda_kernel_traits(
        function_name=function_name,
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
    detected_kernel_type = detect_cuda_kernel_type(
        c_local_matches=c_local_matches,
        fragment_array_elements=fragment_array_elements,
        source=function_source,
        function_name=function_name,
        traits=traits,
        sparse_mask_access_count=sparse_mask_access_count,
        config=config,
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
    traits, evidence = detect_pre_compile_kernel_traits(
        wgmma_shapes=wgmma_shapes,
        op_names=op_names,
        config=config,
    )
    detected_kernel_type = detect_pre_compile_kernel_type(traits)
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
        detected_kernel_traits=merge_kernel_names(ir_info.detected_kernel_traits, source_info.detected_kernel_traits),
        classification_evidence=merge_kernel_names(ir_info.classification_evidence, source_info.classification_evidence),
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
    if violations:
        return AutotuneFilterResult.reject_decision("filter_hard_violation", stage=stage, **details)
    if advisories and filter_config.action == "reject":
        return AutotuneFilterResult.reject_decision("filter_advisory_applied", stage=stage, **details)
    if advisories:
        return AutotuneFilterResult.keep_decision("filter_advisory_report_only", stage=stage, **details)
    return AutotuneFilterResult.keep_decision("filter_targets_passed", stage=stage, **details)


_FILTER_RULE_REGISTRY = AutotuneRuleRegistry(make_default_filter_rules())


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
        if op_name not in PRE_COMPILE_WGMMA_OP_NAMES:
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


def _count_sparse_mask_accesses(source: str) -> int:
    return source.count("BlockMask") + source.count("block_mask") + source.count("blockMask")


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

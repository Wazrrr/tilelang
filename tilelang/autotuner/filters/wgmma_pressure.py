"""Pre-compile WGMMA accumulator register-pressure analysis."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tvm import DataType, tirx
from tvm.tirx.stmt_functor import post_order_visit

WgmmaPressureStatus = Literal[
    "over_budget",
    "within_accumulator_budget",
    "ambiguous",
    "budget_unknown",
    "analysis_unknown",
]
WgmmaPressureConfidence = Literal["exact", "bounded", "lower_bound", "unknown"]


@dataclass(frozen=True)
class WgmmaRegisterPressureInfo:
    """Bounded WGMMA accumulator pressure for one lowered CUDA kernel."""

    lower_bound_registers: int | None
    upper_bound_registers: int | None
    register_budget: int | None
    headroom_lower_bound: int | None
    headroom_upper_bound: int | None
    accumulator_storage_count: int
    wgmma_operation_count: int
    status: WgmmaPressureStatus
    confidence: WgmmaPressureConfidence
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


@dataclass
class _AccumulatorStorage:
    data: Any
    name: str
    fence_counts: list[int] = field(default_factory=list)
    fence_intervals: list[tuple[int, int]] = field(default_factory=list)
    instruction_counts: list[int] = field(default_factory=list)
    instruction_intervals: list[tuple[int, int]] = field(default_factory=list)
    allocation_registers: list[int] = field(default_factory=list)
    unresolved_fence: bool = False
    unresolved_instruction: bool = False
    inconsistent_fence: bool = False

    @property
    def has_fence(self) -> bool:
        return bool(self.fence_counts) or self.unresolved_fence

    def lower_bound(self) -> int | None:
        if self.fence_counts:
            return max(self.fence_counts)
        if self.instruction_counts:
            return max(self.instruction_counts)
        return None

    def upper_bound(self) -> int | None:
        if self.allocation_registers:
            return max(self.allocation_registers)
        if self.fence_counts and not self.unresolved_fence and len(self.fence_intervals) == len(self.fence_counts):
            return _interval_union_size(self.fence_intervals)
        if (
            not self.has_fence
            and self.instruction_counts
            and not self.unresolved_instruction
            and len(self.instruction_intervals) == len(self.instruction_counts)
        ):
            return _interval_union_size(self.instruction_intervals)
        return None

    def validate_fences_against_allocation(self) -> None:
        """Discard fence extents that cannot fit in the lowered local buffer."""
        if not self.allocation_registers or not self.fence_counts:
            return
        allocation_registers = max(self.allocation_registers)
        fence_exceeds_allocation = any(count > allocation_registers for count in self.fence_counts) or any(
            end > allocation_registers for _, end in self.fence_intervals
        )
        if not fence_exceeds_allocation:
            return
        self.inconsistent_fence = True
        self.fence_counts.clear()
        self.fence_intervals.clear()


_WGMMA_C_ARGS: dict[str, tuple[int, int, int]] = {
    # op name: (C dtype, C data, C element offset)
    "tl.ptx_wgmma_ss": (5, 10, 11),
    "tl.ptx_wgmma_rs": (4, 9, 10),
    "tl.ptx_wgmma_sp_ss": (5, 13, 14),
    "tl.ptx_wgmma_sp_rs": (4, 12, 13),
}
_WGMMA_SHAPE_RE = re.compile(r"\bm(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)\b")
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "tf32": "float32",
}
_FENCE_OP = "tl.warpgroup_fence_operand"
_SET_MAX_NREG_OP = "tl.set_max_nreg"
_WARP_SPECIALIZATION_SCOPE = "kWarpSpecializationScope"


def analyze_wgmma_register_pressure(
    device_mod: Any,
    function_name: str,
) -> WgmmaRegisterPressureInfo | None:
    """Analyze WGMMA accumulator pressure without assuming general TIR liveness.

    The lower bound is the largest single accumulator storage that is proven live
    by a fence (or, as a fallback, one WGMMA instruction).  The upper bound sums
    the physical storage assigned to every distinct WGMMA C operand.  This keeps
    multi-accumulator kernels conservative without pretending their live ranges
    are known.
    """
    func = _find_prim_func(device_mod, function_name)
    if func is None or getattr(func, "body", None) is None:
        return None

    nodes = _collect_nodes(func.body)
    wgmma_calls = [node for node in nodes if isinstance(node, tirx.Call) and _op_name(node) in _WGMMA_C_ARGS]
    if not wgmma_calls:
        return None

    storages: list[_AccumulatorStorage] = []
    call_storage_indices: list[tuple[Any, int]] = []
    evidence: list[str] = []
    unresolved_accumulator = False

    for call in wgmma_calls:
        op_name = _op_name(call)
        assert op_name is not None
        dtype_index, data_index, offset_index = _WGMMA_C_ARGS[op_name]
        if len(call.args) <= offset_index:
            unresolved_accumulator = True
            continue
        data = _base_data(call.args[data_index])
        if data is None:
            unresolved_accumulator = True
            continue
        storage = _find_or_add_storage(storages, data)
        call_storage_indices.append((call, _storage_index(storages, storage)))
        dtype_bits = _dtype_bits(call.args[dtype_index])
        register_count = _wgmma_instruction_registers(call, dtype_bits)
        if register_count is None:
            storage.unresolved_instruction = True
            continue
        storage.instruction_counts.append(register_count)
        interval = _register_interval(call.args[offset_index], dtype_bits, register_count)
        if interval is None:
            storage.unresolved_instruction = True
        else:
            storage.instruction_intervals.append(interval)

    for node in nodes:
        if not isinstance(node, tirx.Call) or _op_name(node) != _FENCE_OP or len(node.args) != 4:
            continue
        data = _base_data(node.args[1])
        storage = _find_storage(storages, data)
        if storage is None:
            # Register-sourced A operands are fenced too; only WGMMA C operands count here.
            continue
        dtype_bits = _dtype_bits(node.args[0])
        register_count = _constant_int(node.args[3])
        if register_count is None or register_count < 0:
            storage.unresolved_fence = True
            continue
        storage.fence_counts.append(register_count)
        interval = _register_interval(node.args[2], dtype_bits, register_count)
        if interval is None:
            storage.unresolved_fence = True
        else:
            storage.fence_intervals.append(interval)

    for node in nodes:
        if not isinstance(node, tirx.AllocBuffer):
            continue
        buffer = node.buffer
        storage = _find_storage(storages, buffer.data)
        if storage is None:
            continue
        allocation_registers = _buffer_registers(buffer)
        if allocation_registers is not None:
            storage.allocation_registers.append(allocation_registers)

    for storage in storages:
        storage.validate_fences_against_allocation()

    storage_lowers = [storage.lower_bound() for storage in storages]
    known_lowers = [value for value in storage_lowers if value is not None]
    lower_bound = max(known_lowers) if known_lowers else None

    storage_uppers = [storage.upper_bound() for storage in storages]
    accumulator_paths, paths_collapsed = _accumulator_storage_paths(func.body, call_storage_indices)
    upper_bound = _path_upper_bound(accumulator_paths, storage_uppers)
    register_budget, budget_evidence = _extract_wgmma_role_budget(func.body, wgmma_calls)
    evidence.extend(budget_evidence)
    evidence.extend(
        [
            f"wgmma_operations={len(wgmma_calls)}",
            f"accumulator_storages={len(storages)}",
            f"fenced_storages={sum(storage.has_fence for storage in storages)}",
        ]
    )
    if unresolved_accumulator:
        evidence.append("unresolved_wgmma_accumulator_operand")
    if any(storage.unresolved_fence for storage in storages):
        evidence.append("dynamic_or_invalid_accumulator_fence")
    if any(storage.inconsistent_fence for storage in storages):
        evidence.append("fence_exceeds_static_accumulator_allocation")
    if any(not storage.has_fence for storage in storages):
        evidence.append("instruction_shape_fallback_used")
    if any(value is None for value in storage_uppers):
        evidence.append("accumulator_upper_bound_unknown")
    if len(accumulator_paths) > 1:
        evidence.append(f"accumulator_control_flow_paths={len(accumulator_paths)}")
    if paths_collapsed:
        evidence.append("accumulator_control_flow_paths_collapsed")

    status = _pressure_status(lower_bound, upper_bound, register_budget)
    confidence = _pressure_confidence(
        storages=storages,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unresolved_accumulator=unresolved_accumulator,
    )
    return WgmmaRegisterPressureInfo(
        lower_bound_registers=lower_bound,
        upper_bound_registers=upper_bound,
        register_budget=register_budget,
        headroom_lower_bound=(register_budget - upper_bound if register_budget is not None and upper_bound is not None else None),
        headroom_upper_bound=(register_budget - lower_bound if register_budget is not None and lower_bound is not None else None),
        accumulator_storage_count=len(storages),
        wgmma_operation_count=len(wgmma_calls),
        status=status,
        confidence=confidence,
        evidence=tuple(evidence),
    )


def _find_prim_func(device_mod: Any, function_name: str) -> Any | None:
    for global_var, func in getattr(device_mod, "functions", {}).items():
        attrs = getattr(func, "attrs", None) or {}
        symbol = str(attrs.get("global_symbol", getattr(global_var, "name_hint", str(global_var))))
        if symbol == function_name:
            return func
    return None


def _collect_nodes(stmt: Any) -> list[Any]:
    nodes: list[Any] = []
    post_order_visit(stmt, nodes.append)
    return nodes


def _op_name(call: Any) -> str | None:
    return getattr(getattr(call, "op", None), "name", None)


def _base_data(value: Any) -> Any | None:
    if isinstance(value, tirx.Var):
        return value
    if isinstance(value, tirx.BufferLoad):
        return value.buffer.data
    return None


def _same_object(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return False
    same_as = getattr(lhs, "same_as", None)
    return bool(same_as(rhs)) if same_as is not None else lhs is rhs


def _find_storage(storages: list[_AccumulatorStorage], data: Any) -> _AccumulatorStorage | None:
    for storage in storages:
        if _same_object(storage.data, data):
            return storage
    return None


def _find_or_add_storage(storages: list[_AccumulatorStorage], data: Any) -> _AccumulatorStorage:
    storage = _find_storage(storages, data)
    if storage is not None:
        return storage
    storage = _AccumulatorStorage(data=data, name=str(getattr(data, "name_hint", data)))
    storages.append(storage)
    return storage


def _storage_index(storages: list[_AccumulatorStorage], target: _AccumulatorStorage) -> int:
    for index, storage in enumerate(storages):
        if storage is target:
            return index
    raise ValueError("Accumulator storage is not registered")


def _constant_int(value: Any) -> int | None:
    raw = getattr(value, "value", None)
    if isinstance(raw, int):
        return int(raw)
    if isinstance(value, int):
        return int(value)
    return None


def _string_value(value: Any) -> str | None:
    raw = getattr(value, "value", None)
    if raw is not None:
        return str(raw)
    return value if isinstance(value, str) else None


def _dtype_bits(value: Any) -> int | None:
    text = _string_value(value)
    if text is None:
        text = str(value) if value is not None else None
    if not text:
        return None
    text = _DTYPE_ALIASES.get(text, text)
    try:
        dtype = DataType(text)
    except (TypeError, ValueError):
        return None
    return int(dtype.bits) * int(dtype.lanes)


def _wgmma_instruction_registers(call: Any, dtype_bits: int | None) -> int | None:
    if dtype_bits is None or not call.args:
        return None
    shape_text = _string_value(call.args[0])
    if not shape_text:
        return None
    match = _WGMMA_SHAPE_RE.search(shape_text)
    if match is None:
        return None
    m = int(match.group("m"))
    n = int(match.group("n"))
    # WGMMA distributes every D fragment across one 128-thread warpgroup.
    return (m * n * dtype_bits + 128 * 32 - 1) // (128 * 32)


def _register_interval(
    element_offset: Any,
    dtype_bits: int | None,
    register_count: int,
) -> tuple[int, int] | None:
    offset = _constant_int(element_offset)
    if offset is None or offset < 0 or dtype_bits is None or dtype_bits <= 0:
        return None
    start_bits = offset * dtype_bits
    start_register = start_bits // 32
    end_register = (start_bits + register_count * 32 + 31) // 32
    return (start_register, end_register)


def _buffer_registers(buffer: Any) -> int | None:
    total_elements = 1
    for extent in getattr(buffer, "shape", []):
        value = _constant_int(extent)
        if value is None or value < 0:
            return None
        total_elements *= value
    dtype = getattr(buffer, "dtype", None)
    if dtype is None:
        return None
    try:
        bits = int(dtype.bits) * int(dtype.lanes)
    except (AttributeError, TypeError, ValueError):
        return None
    return (total_elements * bits + 31) // 32


def _interval_union_size(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged_size = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged_size += current_end - current_start
        current_start, current_end = start, end
    return merged_size + current_end - current_start


def _accumulator_storage_paths(
    stmt: Any,
    call_storage_indices: list[tuple[Any, int]],
    *,
    max_paths: int = 64,
) -> tuple[list[frozenset[int]], bool]:
    """Return storage sets used by each structured control-flow path."""

    def visit(node: Any) -> tuple[list[frozenset[int]], bool]:
        if isinstance(node, tirx.IfThenElse):
            then_paths, then_collapsed = visit(node.then_case)
            if node.else_case is None:
                else_paths, else_collapsed = [frozenset()], False
            else:
                else_paths, else_collapsed = visit(node.else_case)
            paths = _deduplicate_paths([*then_paths, *else_paths])
            if len(paths) > max_paths:
                return [_union_paths(paths)], True
            return paths, then_collapsed or else_collapsed

        if isinstance(node, tirx.SeqStmt):
            paths = [frozenset()]
            collapsed = False
            for child in node.seq:
                child_paths, child_collapsed = visit(child)
                paths = _deduplicate_paths([left | right for left in paths for right in child_paths])
                collapsed = collapsed or child_collapsed
                if len(paths) > max_paths:
                    paths = [_union_paths(paths)]
                    collapsed = True
            return paths, collapsed

        child_body = getattr(node, "body", None)
        if child_body is not None:
            return visit(child_body)

        indices = {
            storage_index
            for call, storage_index in call_storage_indices
            if any(_same_object(call, nested) for nested in _collect_nodes(node))
        }
        return [frozenset(indices)], False

    return visit(stmt)


def _deduplicate_paths(paths: list[frozenset[int]]) -> list[frozenset[int]]:
    unique: list[frozenset[int]] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _union_paths(paths: list[frozenset[int]]) -> frozenset[int]:
    result: frozenset[int] = frozenset()
    for path in paths:
        result |= path
    return result


def _path_upper_bound(
    paths: list[frozenset[int]],
    storage_uppers: list[int | None],
) -> int | None:
    path_totals: list[int] = []
    for path in paths:
        values = [storage_uppers[index] for index in path]
        if any(value is None for value in values):
            return None
        path_totals.append(sum(value for value in values if value is not None))
    return max(path_totals, default=0)


def _extract_wgmma_role_budget(
    body: Any,
    wgmma_calls: list[Any],
) -> tuple[int | None, list[str]]:
    scopes = [node for node in _collect_nodes(body) if isinstance(node, tirx.AttrStmt) and str(node.attr_key) == _WARP_SPECIALIZATION_SCOPE]
    if not scopes:
        return None, ["register_budget_unavailable:no_warp_specialization_scope"]

    budget_values: list[int] = []
    covered_calls: list[Any] = []
    missing_budget = False
    evidence: list[str] = []
    for scope in scopes:
        split = _find_warp_specialization_split(scope.body)
        if split is None:
            if any(_is_wgmma_call(node) for node in _collect_nodes(scope.body)):
                missing_budget = True
            continue
        partition = [_constant_int(value) for value in list(scope.node)] if hasattr(scope.node, "__iter__") else []
        if partition and all(value is not None for value in partition):
            evidence.append("warp_specialization_partition=" + "+".join(str(value) for value in partition))
        branches = [split.then_case]
        if split.else_case is not None:
            branches.append(split.else_case)
        for branch in branches:
            branch_nodes = _collect_nodes(branch)
            branch_wgmma = [node for node in branch_nodes if _is_wgmma_call(node)]
            if not branch_wgmma:
                continue
            covered_calls.extend(branch_wgmma)
            set_max_calls = [node for node in branch_nodes if isinstance(node, tirx.Call) and _op_name(node) == _SET_MAX_NREG_OP]
            branch_budgets = []
            for call in set_max_calls:
                if len(call.args) != 2:
                    continue
                register_count = _constant_int(call.args[0])
                is_increase = _constant_int(call.args[1])
                if register_count is not None and is_increase == 1:
                    branch_budgets.append(register_count)
            if not branch_budgets:
                missing_budget = True
            elif len(set(branch_budgets)) == 1:
                budget_values.append(branch_budgets[0])
            else:
                missing_budget = True

    if any(not any(_same_object(call, covered) for covered in covered_calls) for call in wgmma_calls):
        missing_budget = True
        evidence.append("register_budget_unavailable:wgmma_outside_warp_specialization")
    if missing_budget or not budget_values or len(set(budget_values)) != 1:
        evidence.append("register_budget_unavailable:missing_or_conflicting_set_max_nreg")
        return None, evidence
    evidence.append("register_budget_source=set_max_nreg")
    return budget_values[0], evidence


def _find_warp_specialization_split(stmt: Any) -> Any | None:
    current = stmt
    while isinstance(current, tirx.AttrStmt):
        current = current.body
    if isinstance(current, tirx.IfThenElse):
        return current
    if isinstance(current, tirx.SeqStmt):
        candidates = [item for item in current.seq if isinstance(item, tirx.IfThenElse)]
        if len(candidates) == 1:
            return candidates[0]
    candidates = [node for node in _collect_nodes(current) if isinstance(node, tirx.IfThenElse)]
    for candidate in candidates:
        calls = _collect_nodes(candidate)
        if any(isinstance(node, tirx.Call) and _op_name(node) == _SET_MAX_NREG_OP for node in calls):
            return candidate
    return None


def _is_wgmma_call(node: Any) -> bool:
    return isinstance(node, tirx.Call) and _op_name(node) in _WGMMA_C_ARGS


def _pressure_status(
    lower_bound: int | None,
    upper_bound: int | None,
    register_budget: int | None,
) -> WgmmaPressureStatus:
    if lower_bound is None:
        return "analysis_unknown"
    if register_budget is None:
        return "budget_unknown"
    if lower_bound > register_budget:
        return "over_budget"
    if upper_bound is not None and upper_bound <= register_budget:
        return "within_accumulator_budget"
    return "ambiguous"


def _pressure_confidence(
    *,
    storages: list[_AccumulatorStorage],
    lower_bound: int | None,
    upper_bound: int | None,
    unresolved_accumulator: bool,
) -> WgmmaPressureConfidence:
    if lower_bound is None or unresolved_accumulator:
        return "unknown"
    all_fenced = bool(storages) and all(storage.fence_counts for storage in storages)
    if all_fenced and upper_bound == lower_bound:
        return "exact"
    if all_fenced and upper_bound is not None:
        return "bounded"
    return "lower_bound"


__all__ = [
    "WgmmaPressureConfidence",
    "WgmmaPressureStatus",
    "WgmmaRegisterPressureInfo",
    "analyze_wgmma_register_pressure",
]

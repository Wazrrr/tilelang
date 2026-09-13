"""Register storage, proven accumulator demand and resource policy.

Liveness supplies a separate demand estimate. Only established bounds and
resolved physical constraints justify the final rejection decision.
"""

from dataclasses import dataclass, field
from math import ceil, prod
from tvm.ir import Range
from .src.device import target_register_limits
from .src.ir import Region
from .src.ir_utils import _int
from .src.regions import _contains


@dataclass
class RegisterStorage:
    """Report entries plus the subset with per-thread allocation estimates."""

    logical_storage: list[dict]
    modeled_buffers: dict


def analyze_register_storage(col, buffer_facts):
    """Describe each allocation using its scope, dtype and optional layout."""

    storage = []
    modeled = {}
    for buffer in col.buffers:
        facts = buffer_facts[buffer]
        size, dtype = facts.elements, facts.dtype
        entry = {
            "buffer": buffer.name,
            "scope": buffer.scope(),
            "dtype": str(buffer.dtype),
            "logical_elements": size,
            "logical_bits": facts.logical_bits,
            "modeled_registers_per_thread": None,
            "evidence": [],
        }
        layout = buffer_facts[buffer].layout
        if buffer.scope() == "local.fragment" and layout is not None and hasattr(layout, "get_thread_size") and size is not None:
            threads = buffer_facts[buffer].owner_threads
            # An explicit layout fixes ownership. The allocation's index extent
            # remains an estimate: holes and unused elements may be eliminated.
            local_shape = facts.local_shape
            if threads and all(x is not None for x in local_shape):
                slots = prod(local_shape)
                entry["modeled_registers_per_thread"] = {
                    "lower": ceil(slots * dtype.bits * dtype.lanes / 32),
                    "upper": slots * ceil(dtype.bits * dtype.lanes / 32),
                }
                entry["replication"] = buffer_facts[buffer].replication
                entry["computing_threads"] = threads
                entry["evidence"] = [
                    "explicit fragment layout",
                    f"thread extent {threads}",
                    "packing lower bound assumes maximally packed 32-bit registers",
                    "index extent may include holes; allocation alone does not prove liveness",
                ]
                modeled[buffer] = entry
        elif buffer.scope() in ("local", "local.var") and size is not None:
            entry["modeled_registers_per_thread"] = {
                "lower": ceil(size * dtype.bits * dtype.lanes / 32),
                "upper": size * ceil(dtype.bits * dtype.lanes / 32),
            }
            entry["evidence"] = ["thread-private allocation; packing and storage elimination unresolved"]
            modeled[buffer] = entry
        elif buffer.scope() == "local.fragment" and size is not None:
            launch = [_int(v) for k, v in col.threads.items() if k.startswith("threadIdx")]
            if launch and all(x is not None for x in launch):
                entry["balanced_fp32_estimate"] = size / prod(launch) if str(buffer.dtype) == "float32" else None
                entry["evidence"] = ["conditional estimate: balanced layout across all launch threads; ownership not established"]
        storage.append(entry)

    return RegisterStorage(storage, modeled)


@dataclass
class AccumulatorBound:
    """Maximum established bounds across operations; zero means no proof."""

    registers_per_thread: int = 0
    registers_per_block: int = 0
    evidence: list[str] = field(default_factory=list)


def _required_accumulators(col, buffer_facts):
    """Yield (operation, buffer, elements) for full reads and full demands.

    Scalar copies/reductions may stream or fuse away their tiles. Only the
    supported dense MMA accumulator state establishes simultaneous tile demand.
    """

    for op in col.operations:
        if op.unknown or op.predicates or col.unknown or not op.demands:
            continue
        meta = op.metadata
        if meta is None or not hasattr(meta, "cRegion") or bool(getattr(meta, "isTcgen05", False)):
            continue
        seen = set()
        for read in op.reads:
            buffer = read.buffer
            if buffer in seen or buffer.scope() != "local.fragment" or not buffer.same_as(meta.c):
                continue
            seen.add(buffer)
            full = Region(buffer, [Range.from_min_extent(0, x) for x in buffer.shape])
            shape = buffer_facts[buffer].shape
            if not all(x is not None and x > 0 for x in shape):
                continue
            if not _contains(read, full) or not any(_contains(demand, full) for demand in op.demands):
                continue
            yield op, buffer, prod(shape)


def _accumulator_ownership(col, op, buffer, logical_size, modeled_buffers, buffer_facts):
    """Resolve (owner thread bound, replication, evidence), or leave unknown."""

    layout = buffer_facts[buffer].layout
    if layout is None:
        launch = [_int(v) for k, v in op.launch_threads.items() if k.startswith("threadIdx.")]
        if not launch or not all(x is not None and x > 0 for x in launch):
            return None
        # Original launch threads bound the number of accumulator owners from
        # above. Added WS producer threads never enlarge this owner set.
        return (
            prod(launch),
            1,
            [
                "automatic fragment layout: all launch threads bound accumulator ownership from above",
                "replication ignored for the lower bound; no balanced mapping assumed",
            ],
        )

    if buffer not in modeled_buffers:
        return None
    threads = buffer_facts[buffer].owner_threads
    output_size = prod(buffer_facts[buffer].local_shape)
    replication = buffer_facts[buffer].replication
    # Equal volumes alone do not prove distinct physical storage.
    if not replication or output_size * threads != logical_size * replication:
        return None
    try:
        layout.inverse()
    except Exception:
        return None
    return threads, replication, ["explicit invertible fragment layout including replication"]


def analyze_accumulator_bound(col, modeled_buffers, buffer_facts):
    """Establish dtype-aware bounds, taking a maximum across operations.

    Separate operations/branches are never summed, nor are pipeline iterations.
    The block and per-thread maxima are tracked independently.
    """
    result = AccumulatorBound()
    for op, buffer, logical_size in _required_accumulators(col, buffer_facts):
        ownership = _accumulator_ownership(col, op, buffer, logical_size, modeled_buffers, buffer_facts)
        if ownership is None:
            continue
        threads, replication, mapping_evidence = ownership
        dtype = buffer_facts[buffer].dtype
        bits = logical_size * replication * dtype.bits * dtype.lanes
        result.registers_per_block = max(result.registers_per_block, (bits + 31) // 32)
        # Integer ceiling preserves narrow packing and multi-register values.
        bound = (bits + 32 * threads - 1) // (32 * threads)
        if bound > result.registers_per_thread:
            result.registers_per_thread = bound
            result.evidence = [
                f"operation {op.index}: full reads of dense MMA accumulator state",
                *mapping_evidence,
                f"{logical_size} elements × {dtype.bits} bits × {dtype.lanes} lanes × {replication} replication / (32 bits × {threads} threads), rounded up",
                "maximum per-thread demand is at least the CTA average; maximally packed 32-bit registers",
            ]
    return result


def resolve_register_budget(config, target=None):
    """Combine a known target's hardware ceiling with an optional user cap.

    Missing and unrecognized architectures remain unknown. In particular, do
    not infer a cross-compilation target's limits from the local CUDA device.
    This ceiling does not model occupancy or warp-specific register allocation.
    """
    arch, hardware_cap = target_register_limits(target)
    from .targets import resolve_target

    model = resolve_target(target)

    budget = config.register_cap
    source = "user register cap" if budget is not None else None
    if hardware_cap is not None and (budget is None or hardware_cap < budget):
        budget, source = hardware_cap, "architecture register limit"
    return {
        "budget": budget,
        "budget_source": source,
        "hardware_register_cap": hardware_cap,
        "target_arch": arch,
        "target_model": model.to_dict(),
    }


def analyze_register_policy(pressure, config, device_limits=None, *, spill_allowance=0):
    """Resolve physical capacity, then family demand policy, then rejection.

    The family supplies only its soft allowance. Every family obeys the same
    physical register limits and uses the same distinction between estimated
    live demand and a proven accumulator lower bound.
    """
    # 1. Validate the predicted producer/consumer partition and its reservation.
    limits = device_limits or {}
    ws = pressure.get("warp_specialization", {})
    fields = ("producer_threads", "consumer_threads", "producer_register_request", "consumer_register_request")
    known = ws.get("status") == "predicted" and all(type(ws.get(k)) is int and ws[k] > 0 for k in fields)
    reservation = None
    if known:
        reservation = ws["producer_threads"] * ws["producer_register_request"] + ws["consumer_threads"] * ws["consumer_register_request"]
        known = reservation == ws.get("register_reservation_per_block") and ws.get("launch_threads") == (
            ws["producer_threads"] + ws["consumer_threads"]
        )
    if not known:
        reservation = None
    sm_registers = limits.get("registers_per_sm")
    hardware_cap = pressure.get("hardware_register_cap")
    physical_reasons = []
    if reservation is not None and sm_registers is not None and reservation > sm_registers:
        physical_reasons.append(f"policy CTA reservation {reservation} exceeds SM register capacity {sm_registers}")
    if known and hardware_cap is not None and max(ws["producer_register_request"], ws["consumer_register_request"]) > hardware_cap:
        physical_reasons.append(f"policy per-thread register request exceeds hardware limit {hardware_cap}")
    physical = {
        "status": "exceeds_limit" if physical_reasons else "within_limit" if known and sm_registers and hardware_cap else "unknown",
        "precision": "policy_prediction" if known else "unknown",
        "registers_per_block": reservation,
        "registers_per_sm": sm_registers,
        "hardware_register_cap": hardware_cap,
        "resident_blocks_register_limit": sm_registers // reservation if sm_registers and reservation else None,
        "reasons": physical_reasons,
        "assumptions": [
            "physical reservation = producer_threads * producer_request + consumer_threads * consumer_request",
            "compiler initial allocation and allocation granularity still require post-compile validation",
            "logical tile demand and spill allowances never enlarge physical capacity",
        ],
    }
    # 2. Compare logical demand with consumer capacity plus the family margin.
    allowance = spill_allowance
    capacity, basis = pressure.get("budget"), pressure.get("budget_source")
    if known and (capacity is None or ws["consumer_register_request"] <= capacity):
        capacity, basis = ws["consumer_register_request"], "warp-specialization consumer register request"
    live = pressure.get("tile_liveness") or {}
    consumers = ws["consumer_threads"] if known else live.get("computing_threads_estimate")
    tile_regs = live.get("peak_registers_per_block_estimate")
    estimate = (tile_regs + consumers - 1) // consumers if tile_regs is not None and consumers else None
    lower = pressure.get("modeled_lower_bound")
    threshold = capacity + allowance if capacity is not None else None
    excess = max(0, estimate - capacity) if estimate is not None and capacity is not None else None
    proven_exceeds = lower is not None and threshold is not None and lower > threshold
    status = "unknown"
    if excess is not None:
        status = "exceeds_allowance" if excess > allowance else "within_allowance" if excess else "within_capacity"
    demand = {
        "status": status,
        "precision": "estimate" if estimate is not None else "unknown",
        "computing_threads_estimate": consumers,
        "tile_registers_per_block_estimate": tile_regs,
        "registers_per_thread_estimate": estimate,
        "registers_per_thread_lower_bound": lower,
        "capacity_registers_per_thread": capacity,
        "capacity_basis": basis,
        "allowance_registers_per_thread": allowance,
        "soft_limit_registers_per_thread": threshold,
        "excess_registers_per_thread_estimate": excess,
        "excess_bytes_per_thread_estimate": excess * 4 if excess is not None else None,
        "proven_exceeds_allowance": proven_exceeds,
        "assumptions": [
            "dtype packing is included in 32-bit register units; pipeline stages do not replicate accumulators",
            "balanced consumer ownership and conservative tile intervals are estimates; casts may reuse storage",
            "estimated overflow can disappear through compiler reuse; it is not measured spill storage or traffic",
            "only the proven accumulator lower bound can reject on demand; conservative liveness only gates scoring",
            "pipeline timing excludes spill traffic when the score relies on the allowance",
        ],
    }
    # 3. Reject only on physical constraints or a proven demand lower bound.
    reasons = list(physical_reasons)
    if proven_exceeds:
        reasons.append(f"proven tile demand {lower} exceeds demand limit {capacity} + allowance {allowance} = {threshold}")
    return {
        "physical_register_allocation": physical,
        "register_demand": demand,
        "decision": {
            "keep": not reasons or config.mode == "report_only",
            "would_reject": bool(reasons),
            "status": "reject" if reasons else "unknown",
            "reasons": reasons,
        },
    }


def analyze_register_pressure(col, buffer_facts):
    """Describe register demand; the engine resolves capacity and rejection."""
    storage = analyze_register_storage(col, buffer_facts)
    accumulator = analyze_accumulator_bound(col, storage.modeled_buffers, buffer_facts)

    return {
        "logical_storage": storage.logical_storage,
        "modeled_lower_bound": accumulator.registers_per_thread or None,
        "modeled_accumulator_registers_per_block": accumulator.registers_per_block or None,
        "total_register_upper_bound": None,
        "evidence": accumulator.evidence,
        "assumptions": [
            "compiler operand fragments and temporaries are unmodeled",
            "automatic fragment layout is unresolved; recognized warp-specialization policy is reported separately",
            "tile-state bounds are not bounds on total compiler registers",
        ],
    }

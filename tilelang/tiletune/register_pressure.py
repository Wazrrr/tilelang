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


from tiletune_core.register_pressure import analyze_register_policy as analyze_register_policy


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

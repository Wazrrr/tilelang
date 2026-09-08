"""Separate physical register reservations from dtype-aware tile demand.

A spill allowance changes the demand policy, never the SM register file. The
conservative live-tile estimate is useful for ranking, but only an accumulator
lower bound (or a resolved physical constraint) can justify rejection.
"""


def analyze_register_policy(pressure, config, specialization, device_limits=None):
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
    allowance = specialization.register_spill_allowance(config)
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

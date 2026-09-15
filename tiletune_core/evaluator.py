"""Evaluate resolved native engine schedules, including asynchronous completion.

Each region is a compressed repeated DAG. Dependencies name an operation and a
resolved iteration distance (0..32). Storage reuse edges are exported by the
compiler; no buffer-name heuristic or symbolic proof happens here. Sibling
regions drain in order. Independent engines overlap within a region.
"""

from math import ceil
import time

from .contracts import AnalysisReport, Diagnostic
from .schedule import NEG, _delay, _maximum, repeat_transition


def _region_cycles(region, operations, model, concurrent):
    ids = region["operations"]
    count = region["iterations"]
    if type(count) is not int or count < 0 or len(set(ids)) != len(ids):
        raise ValueError("region requires unique operations and resolved nonnegative iterations")
    if count == 0:
        return 0.0, None
    depths = {i: 1 for i in ids}
    for i in ids:
        for dep in operations[i].get("dependencies", []):
            source, distance = dep["operation"], dep.get("distance", 0)
            if source not in depths or type(distance) is not int or not 0 <= distance <= 32:
                raise ValueError("dependency requires a same-region operation and distance in 0..32")
            depths[source] = max(depths[source], distance)
    engines = {engine: i for i, engine in enumerate(model.engines)}
    offsets, size = {}, len(engines)
    for i, depth in depths.items():
        offsets[i], size = size, size + depth
    basis = [[0.0 if i == j else NEG for j in range(size)] for i in range(size)]
    available = {engine: basis[i] for engine, i in engines.items()}
    finished = {}
    for i in ids:
        op = operations[i]
        engine = op["engine"]
        if engine not in engines:
            raise ValueError(f"unsupported native engine {engine}")
        amount = op["work"]
        if type(amount) not in (int, float) or amount < 0:
            raise ValueError("operation work must be resolved and nonnegative")
        rate = model.rates.get(op["service"])
        if amount and not rate:
            return None, Diagnostic("missing_profile", "uncertainty", f"missing service {op['service']}", i)
        latency_key = op.get("completion_latency")
        if latency_key and latency_key not in model.latencies:
            return None, Diagnostic("missing_profile", "uncertainty", f"missing completion latency {latency_key}", i)
        waits = [available[engine]]
        for dep in op.get("dependencies", []):
            source, distance = dep["operation"], dep.get("distance", 0)
            if distance == 0:
                if source not in finished:
                    raise ValueError("zero-distance dependencies must be in topological order")
                waits.append(finished[source])
            else:
                waits.append(basis[offsets[source] + distance - 1])
        issued = _delay(_maximum(*waits), amount * concurrent / rate if amount else 0)
        available[engine] = issued
        finished[i] = _delay(issued, model.latencies.get(latency_key, 0))
    updated = list(available.values())
    for i in ids:
        updated.append(finished[i])
        updated.extend(basis[offsets[i] + d] for d in range(depths[i] - 1))
    state = repeat_transition(tuple(tuple(row) for row in updated), count)
    return max(state, default=0), None


def evaluate(facts, model):
    """Return unknown for incomplete profiles; never fabricate hardware rates."""
    started = time.perf_counter()
    if facts.backend != model.name or any(facts.target.get(k) != v for k, v in model.target.items()):
        raise ValueError("facts and backend profile target do not match")
    diagnostics = list(facts.unresolved)
    allocations, bounds = {}, {}
    for item in facts.storage:
        scope, amount = item["scope"], item["peak"]
        if scope not in model.scopes or type(amount) is not int or amount < 0 or scope in allocations:
            raise ValueError("storage requires one resolved peak per native scope")
        if scope not in model.allocation_units or scope not in model.capacities:
            diagnostics.append(Diagnostic("missing_allocation", "uncertainty", f"missing {scope} capacity or allocation granularity"))
            continue
        unit = model.allocation_units[scope]
        allocated = (amount + unit - 1) // unit * unit
        allocations[scope] = allocated
        if allocated:
            bounds[scope] = model.capacities[scope] // allocated
            if not bounds[scope]:
                diagnostics.append(Diagnostic("resource_violation", "resource", f"{scope} allocation exceeds capacity"))
    if model.units is None or model.max_resident is None:
        diagnostics.append(Diagnostic("missing_residency", "uncertainty", "missing native execution-unit count/residency"))
    grid = facts.launch.get("tasks")
    if type(grid) is not int or grid <= 0:
        diagnostics.append(Diagnostic("unresolved_launch", "uncertainty", "launch task count must be resolved"))
    resident = min([model.max_resident or 0, *bounds.values()])
    score = None
    if not diagnostics:
        concurrent = min(resident, ceil(grid / model.units))
        operations = {op["id"]: op for op in facts.operations}
        if len(operations) != len(facts.operations):
            raise ValueError("duplicate operation identity")
        total = 0.0
        for region in facts.regions:
            cycles, unknown = _region_cycles(region, operations, model, concurrent)
            if unknown:
                diagnostics.append(unknown)
                break
            total += cycles
        if not facts.regions:
            diagnostics.append(Diagnostic("unresolved_schedule", "uncertainty", "no resolved regions"))
        if not diagnostics:
            score = total * ceil(grid / (model.units * concurrent))
    return AnalysisReport(
        model.name,
        score,
        "cycles",
        dict(allocations=allocations, residency_bounds=bounds, resident=resident),
        diagnostics,
        time.perf_counter() - started,
        dict(profile_identity=model.profile_identity, schedule="compressed native engine DAG"),
    )

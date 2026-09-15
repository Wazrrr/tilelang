"""Numerical evaluation of compressed regions exported by compiler adapters."""

from .compute import estimate_phase_cycles


class UnresolvedRegion(ValueError):
    def __init__(self, code, reason):
        self.code = code
        super().__init__(reason)


def region_totals(body):
    totals = dict(
        read_bytes=0, write_bytes=0, **{k: 0 for k in ("gemm_flops", "elementwise_ops", "exp_ops", "reduction_ops", "shared_bytes")}
    )
    for node in body:
        if "operation" in node:
            values = {**node["work"], **{k: node["external_work"][k] for k in ("read_bytes", "write_bytes")}}
            for key, amount in values.items():
                totals[key] = None if amount is None or totals.get(key, 0) is None else totals.get(key, 0) + amount
        else:
            for run in node["runs"]:
                for key, amount in region_totals(run["body"]).items():
                    totals[key] = None if amount is None or totals.get(key, 0) is None else totals.get(key, 0) + run["count"] * amount
    return totals


def estimate_region_cycles(pipeline, concurrent_ctas=1, *, iterations=None):
    from .ampere import schedule_cycles

    if pipeline["unknown"] or not pipeline.get("performance_model"):
        return None
    profile = pipeline["performance_model"]
    phases = {p["operation"]: p for p in pipeline["phases"]}
    rate = profile.get("global_bytes_per_cycle")
    if not rate or any(profile.get(k) is None for k in ("copy_latency_cycles", "barrier_cycles")):
        return None

    def costs(body, producer_ids=()):
        result = {}
        for node in body:
            phase = {**phases[node["operation"]], "work": node["work"]}
            cycles = estimate_phase_cycles(phase, profile, concurrent_ctas) if node["active"] else 0
            if cycles is None:
                raise UnresolvedRegion("unsupported_collective", "incomplete phase service profile")
            work = node["external_work"]
            if node["operation"] not in producer_ids:
                cycles += (work["read_bytes"] + work["write_bytes"]) * concurrent_ctas / rate
                cycles += work["read_groups"] * profile["copy_latency_cycles"]
            result[node["operation"]] = cycles
        return result

    def sequence(body, serial_copies=()):
        cycles = 0
        for node in body:
            if "operation" in node:
                cycles += costs([node])[node["operation"]]
                cycles += sum(c["first_consumer"] == node["operation"] for c in serial_copies) * profile["barrier_cycles"]
            elif node["kind"] == "serial":
                copies = [c for c in pipeline["region_producers"] if c["operation"] in node["operations"]]
                cycles += sum(run["count"] * sequence(run["body"], copies) for run in node["runs"])
            else:
                copies = [c for c in pipeline["region_producers"] if c["operation"] in node["operations"]]
                ids = {c["operation"] for c in copies}
                runs = []
                for run in node["runs"]:
                    work = {p["operation"]: p["external_work"] for p in run["body"]}
                    runs.append(
                        dict(
                            count=run["count"],
                            copies=[{**c, "bytes": work[c["operation"]]["read_bytes"]} for c in copies],
                            costs=costs(run["body"], ids),
                        )
                    )
                value = schedule_cycles(node["plan"], copies, {}, node["iterations"], profile, concurrent_ctas, runs=runs)
                if value is None:
                    raise UnresolvedRegion("unsupported_scheduling", "unresolved independent pipeline service")
                cycles += value
        return cycles

    try:
        variants = pipeline["region_schedule"]["variants"]
        values = [sequence(body) for body in variants] if iterations is None else [sequence(variants[iterations])]
        return dict(
            cycles=max(values), concurrent_ctas=concurrent_ctas, schedule_model="ordered regions; each pipeline drains before its successor"
        )
    except UnresolvedRegion:
        return None

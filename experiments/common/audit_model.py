"""Audit saved comparisons without compiling, timing, or changing rankings.

Counts are candidate-case pairs. Reasons overlap; they must not be summed as
disjoint causes. Compiler comparisons cover only recorded compiled candidates.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from experiments.utils.results import config_key, load_oracle, oracle_at_k, provenance, records_by_index, check_metadata


def read(path):
    return json.loads(path.read_text())


def normalize(reason):
    return re.sub(r"operation \d+", "operation <index>", reason)


def median(values):
    return statistics.median(values) if values else None


def audit_case(path, root, oracle_path=None):
    report = read(path)
    directory = path.parent.parent
    comparison = read(directory / "comparison.json")
    if oracle_path is None:
        reference = comparison.get("oracle")
        if not reference:
            raise ValueError(f"{directory}: no oracle reference; supply --report and --oracle for archived results")
        oracle_path = (directory / reference["path"]).resolve()
        if provenance(oracle_path)["sha256"] != reference["sha256"]:
            raise ValueError(f"{directory}: oracle reference SHA256 mismatch")
    measured = load_oracle(oracle_path)
    check_metadata(measured["path"], path)
    records = report["configs"]
    records_by_index(records)
    if {config_key(c["config"]) for c in records} != set(measured["records"]):
        raise ValueError(f"oracle/config pools differ: {directory}")
    oracle = {c["index"]: measured["records"][config_key(c["config"])] for c in records}
    valid = {i: r["latency_ms"] for i, r in oracle.items() if r["status"] == "benchmarked"}
    selected = report["selection"]["selected_indices"]
    eligible = [c["index"] for c in records if c["ranking"]["tier"] == "eligible"]
    best = min(valid, key=valid.get) if valid else None
    best_eligible = min((i for i in eligible if i in valid), key=valid.get, default=None)
    best_selected = min((i for i in selected if i in valid), key=valid.get, default=None)
    reason_counts = Counter()
    tiers = Counter()
    candidates = []
    compiler = []
    parallel_work = []
    for record in records:
        i = record["index"]
        modules = record["modules"]
        reasons = {name: value.get("unknown", []) for name, value in modules.items() if isinstance(value, dict) and value.get("unknown")}
        if record["pre_lowering"].get("reasons"):
            reasons["pre_lowering"] = record["pre_lowering"]["reasons"]
        unique = {normalize(reason) for entries in reasons.values() for reason in entries}
        reason_counts.update(unique)
        tier = record["ranking"]["tier"]
        tiers[tier] += 1
        candidates.append(
            dict(
                index=i,
                config=record["config"],
                tier=tier,
                selected=i in selected,
                oracle_status=oracle[i]["status"],
                oracle_latency_ms=valid.get(i),
                reasons=reasons,
                demand_status=record["pressure"]["register_demand"]["status"],
            )
        )
        operations = {op["index"]: op for op in record["tile_propagation"]["operations"]}
        for phase in modules["pipeline_overlap"]["phases"]:
            if phase["kind"] != "elementwise":
                continue
            axes = [axis["extent"] for axis in operations[phase["operation"]]["loops"] if axis["kind"] == "1"]
            try:
                elements = math.prod(int(extent) for extent in axes)
            except ValueError:
                elements = None
            parallel_work.append(dict(index=i, operation=phase["operation"], parallel_elements=elements, recorded_work=phase["work"]))
        resources = record.get("compiler_resources") or {}
        waves = modules["waves"]
        threads = waves.get("launch_threads")
        block_registers = waves.get("registers_per_block_estimate")
        # A single entry is a direct comparison. Multiple native kernels may
        # have different launch domains, so do not combine their allocations.
        if len(resources) == 1 and threads and block_registers:
            observed = next(iter(resources.values()))
            registers = observed.get("registers")
            if registers:
                bounds = dict(waves["resident_blocks_limits"])
                capacity = waves["device_limits"].get("registers_per_sm")
                if capacity:
                    bounds["registers"] = capacity // (registers * threads)
                compiler.append(
                    dict(
                        index=i,
                        config=record["config"],
                        proxy_registers_per_thread=block_registers / threads,
                        compiler_registers_per_thread=registers,
                        compiler_over_proxy=registers * threads / block_registers,
                        predicted_resident_ctas=waves["resident_blocks_per_sm_estimate"],
                        resident_upper_bound_with_compiler_registers=min(bounds.values()) if capacity else None,
                        compiler_resources=observed,
                    )
                )
    best_ms = valid.get(best)
    eligible_ms = valid.get(best_eligible)
    selected_ms = valid.get(best_selected)
    return dict(
        workload=comparison["workload"]["name"],
        operation=comparison["workload"]["op"],
        device=comparison["device"],
        report=str(path.relative_to(root)),
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        analysis_version=report["analysis_version"],
        grid=len(records),
        oracle_valid=len(valid),
        tiers=dict(tiers),
        reason_counts=dict(reason_counts),
        selected_indices=selected,
        finite_but_compile_failed=[i for i in eligible if oracle[i]["status"] == "compilation_failed"],
        winner=best,
        winner_config=oracle[best]["config"] if best is not None else None,
        winner_eligible=best in eligible,
        best_eligible=best_eligible,
        best_selected=best_selected,
        coverage_ceiling=best_ms / eligible_ms if eligible_ms else None,
        ranking_within_eligible=eligible_ms / selected_ms if eligible_ms and selected_ms else None,
        oracle_at_k=oracle_at_k(valid, selected),
        oracle_sources=measured["sources"],
        demand_excluded_but_valid=[c["index"] for c in candidates if c["demand_status"] == "exceeds_allowance" and c["index"] in valid],
        compiler_comparisons=compiler,
        parallel_work=parallel_work,
        candidates=candidates,
    )


def audit(root, *, report_path=None, oracle_path=None):
    paths = (
        [report_path]
        if report_path
        else sorted(p for p in root.rglob("tiletune/tiletune.json") if (p.parent.parent / "comparison.json").is_file())
    )
    if not paths:
        raise ValueError(f"no saved TileTune test reports under {root}")
    cases = [audit_case(path, root, oracle_path) for path in paths]
    reasons = Counter()
    tiers = Counter()
    for case in cases:
        reasons.update(case["reason_counts"])
        tiers.update(case["tiers"])
    compiler = [row for case in cases for row in case["compiler_comparisons"]]
    occupancy = [row for row in compiler if row["resident_upper_bound_with_compiler_registers"] is not None]
    spills = [
        row
        for row in compiler
        if any(row["compiler_resources"].get(key, 0) for key in ("spill_stores_bytes", "spill_loads_bytes", "local_bytes"))
    ]
    return dict(
        version=2,
        root=str(root.resolve()),
        semantics={
            "counts": "candidate-case pairs; normalized reasons deduplicated per candidate; reasons overlap",
            "coverage_ceiling": "oracle latency / fastest valid eligible latency",
            "ranking_within_eligible": "fastest valid eligible latency / fastest valid frozen-shortlist latency",
            "compiler": "selected compiled candidates only; register-only occupancy substitution ignores allocation granularity",
            "parallel_work": "logical parallel domain and saved work; interpretation depends on audited source version",
            "mutation": "no model changes, no reranking, no new GPU timings",
        },
        summary=dict(
            cases=len(cases),
            candidates=sum(c["grid"] for c in cases),
            valid_candidates=sum(c["oracle_valid"] for c in cases),
            tiers=dict(tiers),
            reason_counts=dict(reasons),
            eligible_oracle_winners=sum(c["winner_eligible"] for c in cases),
            finite_but_compile_failed=sum(len(c["finite_but_compile_failed"]) for c in cases),
            demand_excluded_but_valid=sum(len(c["demand_excluded_but_valid"]) for c in cases),
            compiler_pairs=len(compiler),
            median_compiler_over_proxy=median([c["compiler_over_proxy"] for c in compiler]),
            occupancy_overestimates=sum(
                c["predicted_resident_ctas"] > c["resident_upper_bound_with_compiler_registers"] for c in occupancy
            ),
            occupancy_comparisons=len(occupancy),
            compiled_with_spill_or_local=len(spills),
        ),
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="completed study/comparison output directory")
    parser.add_argument("--output", type=Path, required=True, help="new audit JSON; input reports are read only")
    parser.add_argument("--report", type=Path, help="one saved TileTune report, for an archived run without an oracle reference")
    parser.add_argument("--oracle", type=Path, help="explicit oracle for --report")
    args = parser.parse_args()
    if bool(args.report) != bool(args.oracle):
        parser.error("--report and --oracle must be supplied together")
    result = audit(
        args.root.resolve(),
        report_path=args.report.resolve() if args.report else None,
        oracle_path=args.oracle.resolve() if args.oracle else None,
    )
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

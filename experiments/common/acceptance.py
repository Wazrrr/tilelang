"""Acceptance gates retain missing targets, cases, failures, and all seeds."""

import json
import math
from pathlib import Path
from statistics import median


def assess_seed(cases, comparisons):
    by_name = {row["workload"]["name"]: row for row in comparisons}
    details, ratios, speedups = [], [], []
    for case in cases:
        row = by_name.get(case["name"], {})
        methods = row.get("methods", {})
        tuned, oracle = methods.get("tiletune", {}), methods.get("brute_force", {})
        diag = row.get("diagnostics", {}).get("tiletune", {})
        validation = row.get("validation") or {}
        samples = validation.get("tiletune", {}).get("samples_ms", [])
        correct = tuned.get("correctness") == "passed" and tuned.get("status") == "completed"
        correct &= len(samples) == 7 and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in samples)
        coverage = diag.get("correct_score_coverage")
        ratio = diag.get("curves", {}).get("20", {}).get("oracle_at_k")
        online, exhaustive = tuned.get("tuning_seconds"), oracle.get("tuning_seconds")
        speedup = exhaustive / online if online and exhaustive and online > 0 and exhaustive > 0 else None
        if ratio is not None:
            ratios.append(ratio)
        if speedup is not None:
            speedups.append(speedup)
        details.append(
            dict(
                name=case["name"],
                family=case["op"],
                correct_winner=bool(correct),
                correct_score_coverage=coverage,
                coverage_pass=coverage is not None and coverage >= 0.9,
                oracle_at_20=ratio,
                online_speedup=speedup,
                status=tuned.get("status", "missing"),
                reason=tuned.get("reason"),
                failures=tuned.get("candidate_statuses"),
                diagnostics=diag,
            )
        )
    all_present = (
        bool(cases) and set(by_name) == {c["name"] for c in cases} and all(r["correct_winner"] and r["coverage_pass"] for r in details)
    )
    quality = median(ratios) if len(ratios) == len(cases) else None
    speed = median(speedups) if len(speedups) == len(cases) else None
    return dict(
        accepted=bool(all_present and quality is not None and quality >= 0.95 and speed is not None and speed > 1),
        cases=details,
        median_oracle_at_20=quality,
        median_online_speedup=speed,
        worst_oracle_at_20=min(ratios) if ratios else None,
        families={op: [r for r in details if r["family"] == op] for op in sorted({c["op"] for c in cases})},
    )


def aggregate_study(plan, root):
    root = Path(root)
    baseline_refs = json.loads((root / "baselines.json").read_text()) if (root / "baselines.json").exists() else {}
    targets = {}
    case_count = len(plan["splits"]["test"])
    for device in plan["devices"]:
        name = device["name"]
        seeds = {}
        for seed in plan["budget"]["seeds"]:
            base = root / str(seed) / name
            path = base / "comparison.json"
            comparisons = json.loads(path.read_text())["results"] if path.exists() else []
            seeds[str(seed)] = assess_seed(plan["splits"]["test"], comparisons)
            # Preparation records remain separate from online candidate tuning.
            preparation = []
            for p in sorted(base.rglob("profile-preparation.json")):
                preparation.append(dict(path=str(p.relative_to(root)), **json.loads(p.read_text())))
            profile = root / "preparation" / name / "result.json"
            if profile.exists():
                profile_result = json.loads(profile.read_text())
                preparation.append(
                    dict(
                        path=str(profile.relative_to(root)),
                        seconds=profile_result.get("preparation_seconds"),
                        status=profile_result["status"],
                    )
                )
            models = []
            model_paths = set(base.rglob("models/xgboost-*.json"))
            for reference in baseline_refs.get(name, {}).values():
                model_paths.update((Path(reference["path"]) / "collection" / name / "models").glob("xgboost-*.json"))
            for path in sorted(model_paths):
                model = json.loads(path.read_text())
                training = model.get("training", {})
                fields = ("fit_seconds", "training_collection_seconds", "validation_collection_seconds")
                models.append(dict(path=str(path.resolve()), costs_seconds={k: training.get(k) for k in fields}))
            seeds[str(seed)]["preparation"] = preparation
            seeds[str(seed)]["training"] = models
            profile_seconds = (
                sum(p["seconds"] for p in preparation) if preparation and all(p.get("seconds") is not None for p in preparation) else None
            )
            costs = {}
            for method in plan.get("methods", ("tiletune", "carver", "xgboost", "brute_force")):
                online = [(r.get("methods", {}).get(method) or {}).get("tuning_seconds") for r in comparisons]
                online_total = sum(online) if len(online) == case_count and all(v is not None for v in online) else None
                model_costs = [v for m in models for v in m["costs_seconds"].values()]
                training_total = sum(model_costs) if models and all(v is not None for v in model_costs) else None
                prep = profile_seconds if method == "tiletune" else training_total if method == "xgboost" else 0
                costs[method] = dict(
                    measurement_origin="baseline bundle"
                    if method in ("carver", "xgboost", "brute_force") and name in baseline_refs
                    else "this run",
                    online_seconds=online_total,
                    preparation_seconds=prep,
                    first_use_seconds=online_total + prep if online_total is not None and prep is not None else None,
                    amortized_seconds_per_case=(online_total + prep) / case_count
                    if case_count and online_total is not None and prep is not None
                    else None,
                )
            seeds[str(seed)]["costs"] = costs
        targets[name] = dict(
            accepted=bool(seeds) and all(s["accepted"] for s in seeds.values()), seeds=seeds, unavailable=plan["unavailable"].get(name)
        )
    complete_matrix = set(targets) == {"ampere", "hopper", "blackwell", "mi355x", "ascend910b"}
    complete_families = case_count == 25 and {c["op"] for c in plan["splits"]["test"]} == {
        "gemm",
        "attention",
        "kda_chunk_o",
        "gemm_fp8",
        "grouped_gemm",
    }
    return dict(
        version=1,
        suite=plan["suite"],
        accepted=bool(targets)
        and all(t["accepted"] for t in targets.values())
        and (complete_matrix or plan["suite"] != "final" or not complete_families),
        scope="matrix" if complete_families else "families",
        complete_family_matrix=complete_families,
        five_target_accepted=plan["suite"] == "final"
        and complete_matrix
        and complete_families
        and all(t["accepted"] for t in targets.values()),
        complete_target_matrix=complete_matrix,
        targets=targets,
        note="Missing cases and unknown costs fail gates; medians are reported independently per seed.",
    )

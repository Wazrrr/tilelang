"""Collect immutable baselines once, then evaluate each TileTune revision."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path

from .run import make_request

from experiments.utils.io import write_json
from .spec import Device, Workload
from experiments.utils.baseline_store import collect_bundle, identities, runtime_identity
from experiments.utils.results import read
from experiments.families import FAMILIES


def execute(plan, output, settings, *, baseline_root, baseline_seed=123):
    from experiments.suite import _existing_or_run
    from experiments.compare_results import compare
    from experiments.utils.diagnostics import assess
    from .acceptance import aggregate_study

    references = {}
    for description in plan["devices"]:
        device = Device(**description)
        if device.name in plan["unavailable"]:
            continue
        runtime = runtime_identity(device)
        # Primitive profiles affect TileTune only, and never baseline identity.
        baseline_device = replace(device, profiles=None, performance_model=None)
        bundles, measurements = {}, {}
        references[device.name] = {}
        for op in dict.fromkeys(w["op"] for w in plan["splits"]["test"]):
            family_plan = dict(plan, splits={split: [w for w in values if w["op"] == op] for split, values in plan["splits"].items()})
            names = {w["name"] for values in family_plan["splits"].values() for w in values}
            family_device = replace(
                baseline_device,
                configs={k: v for k, v in (baseline_device.configs or {}).items() if k in names} or None,
                subsets={k: v for k, v in (baseline_device.subsets or {}).items() if k in names} or None,
            )
            measurement, identity = identities(family_plan, device, settings, runtime, baseline_seed)
            bundle, reused = collect_bundle(
                Path(baseline_root) / device.name / FAMILIES[op], identity, measurement, family_device, settings
            )
            bundles[op], measurements[op] = bundle, measurement
            references[device.name][op] = dict(
                path=str(bundle),
                reused=reused,
                seed=baseline_seed,
                manifest_sha256=hashlib.sha256((bundle / "complete.json").read_bytes()).hexdigest(),
            )
        write_json(output / "baselines.json", references)
        if not device.profiles and not device.performance_model:
            profiles = {}
            for item in plan["splits"]["test"]:
                workload = Workload(**item)
                if workload.dtype in profiles:
                    continue
                preparation = output / "preparation" / device.name / workload.dtype
                request = make_request(workload, device, dict(settings, method="profile", memory_regime="streaming"))
                result = _existing_or_run(request, preparation)
                if result["status"] != "profiled":
                    raise RuntimeError(f"TileTune profile preparation failed for {workload.dtype}: {result}")
                profiles.update(result["profiles"])
            device = replace(device, profiles=profiles)
        for seed in plan["budget"]["seeds"]:
            root = output / str(seed) / device.name
            root.mkdir(parents=True, exist_ok=True)
            comparisons = []
            for item in plan["splits"]["test"]:
                w = Workload(**item)
                bundle, measurement = bundles[w.op], measurements[w.op]
                case = root / device.name / "test" / w.name
                case.mkdir(parents=True, exist_ok=True)
                baseline_case = bundle / "collection" / device.name / "test" / w.name
                options = dict(
                    settings,
                    method="top_k",
                    metric="pipeline_time",
                    top_k=plan["top_k"],
                    seed=123,
                    selection_seed=seed,
                    memory_regime="streaming",
                    trace=False,
                    measurement_identity=measurement,
                )
                request = make_request(w, device, options)
                result = _existing_or_run(request, case / "tiletune")
                if result["status"] == "failed":
                    raise RuntimeError(f"TileTune worker failed: {case}: {result}")
                methods = dict(read(baseline_case / "methods.json"), tiletune=result)
                reports = {}
                for method in ("carver", "xgboost", "brute_force"):
                    link = case / method
                    if not link.exists():
                        link.symlink_to(baseline_case / method, target_is_directory=True)
                    elif link.resolve() != (baseline_case / method).resolve():
                        raise ValueError(f"baseline reference changed: {link}")
                    path = link / (method + ".json")
                    if path.exists():
                        reports[method] = read(path)
                tiletune_path = case / "tiletune/tiletune.json"
                if tiletune_path.exists():
                    reports["tiletune"] = read(tiletune_path)
                    write_json(
                        case / "frozen-rankings.json",
                        dict(
                            methods={
                                "tiletune": {"ranking": reports["tiletune"]["ranking"], "selection": reports["tiletune"].get("selection")}
                            }
                        ),
                    )
                oracle_path = baseline_case / "brute_force/outcomes.json"
                oracle = read(oracle_path)
                diagnostics = {name: assess(report, oracle, seed=seed) for name, report in reports.items() if name != "brute_force"}
                # Only the new TileTune winner is remeasured. Baseline timings
                # and costs remain the original recorded measurements.
                validation = None
                if result["status"] == "completed":
                    validation_request = make_request(
                        w,
                        device,
                        dict(options, method="remeasure", methods={"tiletune": result}, validation_repeats=plan["validation_repeats"]),
                    )
                    checked = _existing_or_run(validation_request, case / "winner-remeasurement")
                    if checked["status"] != "remeasured":
                        raise RuntimeError(f"TileTune winner remeasurement failed: {checked}")
                    validation = checked["validation"]
                write_json(case / "methods.json", methods)
                row = dict(
                    workload=w.to_dict(),
                    device=device.name,
                    methods=methods,
                    diagnostics=diagnostics,
                    validation=validation,
                    oracle=dict(path=os.path.relpath(oracle_path, case), sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest()),
                    baseline_bundle=references[device.name][w.op],
                )
                write_json(case / "comparison.json", row)
                paths = {name: case / name for name in ("carver", "xgboost", "tiletune")}
                # Unsupported Carver cases carry result.json with the reason.
                curves = compare(oracle_path, paths, sorted({1, 5, 10, 20, 50, plan["top_k"]}))
                write_json(case / "oracle-curves.json", curves)
                comparisons.append(row)
                write_json(root / "comparison.json", dict(version=2, results=comparisons))
    report = aggregate_study(plan, output)
    report["baselines"] = references
    write_json(output / "acceptance.json", report)
    return int(not report["accepted"])

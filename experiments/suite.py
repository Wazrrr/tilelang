"""One family definition, three deterministic budgets, five explicit targets.

Planning and freezing need only Python. Execution reuses isolated workers and
immutable baselines independently of TileTune revisions and repeat seeds.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from experiments.families import DEFAULT_OPS, FAMILIES, family_module
from tiletune_core.contracts import digest
from experiments.common.run import make_request, run_case
from experiments.utils.io import write_json
from experiments.common.spec import Device, TARGETS, configuration_space
from experiments.utils.subsets import pairwise_subset
from experiments.common.spaces import PRESETS

CORE_OPS = DEFAULT_OPS
CORE_FAMILIES = tuple(dict.fromkeys(FAMILIES[op] for op in CORE_OPS))
CORE_TARGETS = ("ampere", "hopper", "blackwell", "mi355x", "ascend910b")
BUDGETS = {
    "smoke": dict(cases=5, configurations=16, seeds=[123], compare=False),
    "development": dict(cases=25, configurations=256, seeds=[123], compare=True),
    "final": dict(cases=25, configurations=None, seeds=[123, 456, 789], compare=True),
    # A complete expanded-pool benchmark can precede the development gates.
    # It uses the final shapes/protocol without claiming final acceptance.
    "full": dict(cases=25, configurations=None, seeds=[123, 456, 789], compare=True),
}
DEVICE_PATTERNS = dict(ampere="A100", hopper="H200", blackwell="B200|GB200", mi355x="MI355X", ascend910b="910B|A2")


def core_cases(suite, families=None):
    if suite not in BUDGETS:
        raise ValueError(f"unknown suite {suite}")
    families = CORE_FAMILIES if families is None else tuple(families)
    if not families or len(set(families)) != len(families) or set(families) - set(FAMILIES.values()):
        raise ValueError(f"families must be unique names from {tuple(FAMILIES.values())}")
    ops = [op for op in FAMILIES if FAMILIES[op] in families]
    cases = [w for op in ops for w in family_module(op, "cases").cases(holdout=suite in ("full", "final"))]
    if suite in ("full", "final"):
        frozen = json.loads(Path(__file__).with_name("manifests").joinpath("five_target_final.json").read_text())["workloads"]
        frozen = [w for w in frozen if w["op"] in ops]
        if [w.to_dict() for w in cases] != frozen:
            raise ValueError("final family definitions differ from the frozen holdout manifest")
    if suite == "smoke":
        seen = set()
        smoke = []
        for workload in cases:
            if workload.op not in seen:
                seen.add(workload.op)
                smoke.append(workload)
        return smoke
    return cases


def study_plan(suite, devices=None, *, families=None, config_space=None):
    tests = core_cases(suite, families)
    families = list(dict.fromkeys(FAMILIES[op] for op in FAMILIES if any(w.op == op for w in tests)))
    if suite in ("final", "full") and config_space is not None and any(w.config_space != config_space for w in tests):
        raise ValueError("final/full suites require each family's frozen configuration space")
    budget = dict(BUDGETS[suite], cases=len(tests))
    devices = devices or [Device(name, TARGETS[name], expected_device_pattern=DEVICE_PATTERNS[name]) for name in CORE_TARGETS]
    splits = dict(train=[], validation=[], test=tests)
    if budget["compare"]:
        for op in FAMILIES:
            if FAMILIES[op] not in families:
                continue
            a, b, validation = family_module(op, "cases").training_cases()
            splits["train"].extend((a, b))
            splits["validation"].append(validation)
    if config_space:
        splits = {split: [replace(w, config_space=config_space) for w in cases] for split, cases in splits.items()}
    all_cases = [w for values in splits.values() for w in values]
    planned, audits, availability = [], {}, {}
    for device in devices:
        subsets, audit = {}, {}
        for w in all_cases:
            # Never substitute a CUDA knob domain for Cube/Vector schedules.
            if device.target["kind"] not in ("cuda", "hip") and not (device.configs and w.name in device.configs):
                availability[device.name] = "native family configuration grids and pinned Ascend worker required"
                continue
            space = configuration_space(w, device)
            audit[w.name] = pairwise_subset(w, device, space["configs"], budget["configurations"])
            subsets[w.name] = audit[w.name]["indices"]
        planned.append(replace(device, subsets=subsets or None))
        audits[device.name] = audit
    return dict(
        version=1,
        suite=suite,
        families=families,
        budget=budget,
        metric="pipeline_time",
        top_k=20,
        dtypes=sorted({w.dtype for w in tests}),
        devices=[d.to_dict() for d in planned],
        splits={k: [w.to_dict() for w in v] for k, v in splits.items()},
        subsets=audits,
        unavailable=availability,
        methods=["brute_force", "carver", "xgboost", "tiletune"],
        validation_repeats=7,
        memory_regime="streaming",
        failure_policy="attempts without replacement",
        xgboost=dict(
            training_shapes_per_family=2,
            validation_shapes_per_family=1,
            sample_fraction=0.1,
            rounds=600,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            patience=20,
        ),
    )


def freeze(plan, output, settings):
    from experiments.utils.cli import source_hashes

    profiles = {}
    for device in plan["devices"]:
        for path in (device.get("profiles") or {}).values():
            profiles[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = dict(plan=plan, settings=settings, sources=source_hashes("experiments/suite.py"), profiles=profiles)
    lock = dict(version=1, sha256=digest(payload), **payload)
    path = output / "study-lock.json"
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise ValueError("frozen study inputs changed; create a new study directory")
    else:
        write_json(path, lock)
    return lock


def _existing_or_run(request, output):
    from experiments.common.run import validate_result

    if (output / "result.json").exists():
        if json.loads((output / "request.json").read_text()) != request:
            raise ValueError("resumed case request differs from the frozen request")
        result = validate_result(json.loads((output / "result.json").read_text()), request)
        monitor = output / "monitor.json"
        retry = (
            result["status"] == "failed"
            and monitor.exists()
            and json.loads(monitor.read_text())["status"] in ("contended", "timeout", "monitor_gap", "host_contended")
        )
        if not retry:
            return result
    if output.exists():
        from datetime import datetime, timezone

        output.rename(output.with_name(output.name + ".interrupted." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")))
    return run_case(request, output)


def execute_smoke(plan, output, settings):
    from experiments.common.spec import Workload
    from experiments.common.comparison import wait_for_idle
    from experiments.utils.monitor import cuda_device

    results = []
    for description in plan["devices"]:
        device = Device(**description)
        with cuda_device(device):
            for item in plan["splits"]["test"]:
                workload = Workload(**item)
                if device.name in plan["unavailable"]:
                    result = dict(device=device.name, workload=workload.name, status="unavailable", reason=plan["unavailable"][device.name])
                else:
                    if not device.worker:
                        wait_for_idle(output)
                    request = make_request(
                        workload,
                        device,
                        dict(settings, method="smoke", metric="pipeline_time", top_k=20, seed=123, trace=False, memory_regime="streaming"),
                    )
                    result = _existing_or_run(request, output / device.name / workload.name)
                results.append(result)
                write_json(output / "smoke.json", dict(version=1, results=results))
    return int(any(r["status"] != "smoke_passed" for r in results))


def execute_comparison(plan, output, settings, *, baseline_root=None, baseline_seed=123, run_baselines=False):
    from experiments.common.study import execute

    return execute(plan, output, settings, baseline_root=baseline_root, baseline_seed=baseline_seed, run_baselines=run_baselines)


def main(argv=None, *, family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=BUDGETS, default="development" if family else "smoke")
    parser.add_argument("--devices", "--device", nargs="+", choices=CORE_TARGETS, default=["ampere"] if family else list(CORE_TARGETS))
    if family:
        parser.set_defaults(families=[family])
    else:
        parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES.values()), default=list(CORE_FAMILIES))
    parser.add_argument(
        "--config-space",
        choices=PRESETS,
        help="Core families use expanded; final/full use the complete frozen pools",
    )
    parser.add_argument("--device-manifest", type=Path, help="JSON list of explicit Device objects, including external worker argv")
    parser.add_argument("--output", type=Path, help="Run output; family commands default to their own results directory")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Override family-owned GPU-specific baseline storage (ROOT/TARGET/FAMILY)",
    )
    parser.add_argument("--run-baselines", action="store_true", help="Explicitly collect/refresh baselines only; do not run TileTune")
    parser.add_argument(
        "--baseline-seed", type=int, default=123, help="Fixed XGBoost collection/training seed, independent of TileTune repeats"
    )
    parser.add_argument("--top-k", type=int, default=20, help="TileTune online budget; final acceptance uses K=20")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--freeze", action="store_true", help="freeze reviewable inputs without executing the study")
    parser.add_argument("--development-report", type=Path, help="passing development acceptance.json required for final execution")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--case-timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    devices = (
        [Device(**d) for d in json.loads(args.device_manifest.read_text())]
        if args.device_manifest
        else [Device(name, TARGETS[name], expected_device_pattern=DEVICE_PATTERNS[name]) for name in args.devices]
    )
    try:
        plan = study_plan(args.suite, devices, families=args.families, config_space=args.config_space)
    except ValueError as error:
        parser.error(str(error))
    if args.top_k < 1 or args.baseline_seed < 0:
        parser.error("positive top-k and nonnegative baseline seed required")
    if args.suite == "final" and args.top_k != 20:
        parser.error("final acceptance uses K=20; use --suite full for other online budgets")
    if args.run_baselines and args.suite == "smoke":
        parser.error("--run-baselines requires a comparison suite, such as --suite full")
    baseline_root = args.baseline_root.resolve() if args.baseline_root else None
    plan.update(
        top_k=args.top_k,
        baseline_seed=args.baseline_seed,
        baseline_root=str(baseline_root) if baseline_root else None,
        baseline_action="collect" if args.run_baselines else "read_only",
    )
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    settings = {k: getattr(args, k) for k in ("workers", "warmup", "rep", "timeout", "case_timeout")}
    if any(v <= 0 for v in settings.values()):
        parser.error("execution budgets must be positive")
    if args.output is None:
        from datetime import datetime, timezone

        root = Path("experiments") / args.families[0] / "results" if len(args.families) == 1 else Path("experiments/results/studies")
        category = "baseline-collection" if args.run_baselines else "tiletune"
        args.output = root / category / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output.resolve()
    freeze(plan, output, settings)
    if args.freeze:
        return 0
    if args.suite == "final" and not args.run_baselines:
        gate = json.loads(args.development_report.read_text()) if args.development_report else {}
        if (
            gate.get("suite") != "development"
            or not gate.get("accepted")
            or any(not gate.get("targets", {}).get(d.name, {}).get("accepted") for d in devices)
            or any(
                not {w["name"] for w in plan["splits"]["test"]} <= {c["name"] for c in seed.get("cases", [])}
                for d in devices
                for seed in gate.get("targets", {}).get(d.name, {}).get("seeds", {}).values()
            )
            or any(not gate.get("targets", {}).get(d.name, {}).get("seeds") for d in devices)
        ):
            parser.error("final execution requires passing development gates for every requested target")
    return (
        execute_smoke(plan, output, settings)
        if args.suite == "smoke"
        else execute_comparison(
            plan, output, settings, baseline_root=baseline_root, baseline_seed=args.baseline_seed, run_baselines=args.run_baselines
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

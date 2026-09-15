"""Freeze every seed's selections before collecting a shared held-out oracle.

The supplied split manifest retains exact tail dimensions. Each seed/policy
collects its own 10% attempted training and validation samples. Oracle timings
are shared only after every model and shortlist has been frozen.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .run import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "manifests" / "repair_study_ampere.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    root = args.output.resolve()
    methods = ["tiletune", "tiletune_exploration", "xgboost", "xgboost_stratified", "random"]
    plan = dict(
        version=1,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        seeds=args.seeds,
        methods=methods,
        top_k=20,
        sample_fraction=0.1,
        input_seed=123,
        winner_rounds=7,
        workers=args.workers,
    )
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=args.resume)
    if (root / "plan.json").exists() and json.loads((root / "plan.json").read_text()) != plan:
        raise ValueError("study resume differs from frozen plan")
    write_json(root / "plan.json", plan)
    manifest = json.loads(args.manifest.read_text())
    profiled_manifest = root / "splits-profiled.json"
    if not profiled_manifest.exists():
        from .comparison import wait_for_idle

        wait_for_idle(root, wait=True)
        started = time.perf_counter()
        profile = root / "primitive-profile.json"
        dtypes = sorted({w.get("dtype", "float16") for values in manifest["splits"].values() for w in values})
        # The profiling process must release its CUDA context before child
        # comparison workers enforce GPU isolation.
        code = (
            "import sys; from tilelang.tiletune import profile_device; "
            "[profile_device(input_dtype=d, cache_path=sys.argv[1], memory_regime='streaming') for d in sys.argv[2:]]"
        )
        with (root / "profile.log").open("a") as stream:
            subprocess.run([sys.executable, "-c", code, str(profile), *dtypes], stdout=stream, stderr=subprocess.STDOUT, check=True)
        for device in manifest["devices"]:
            device["profiles"] = {dtype: str(profile) for dtype in dtypes}
        write_json(profiled_manifest, manifest)
        write_json(
            root / "profile-preparation.json",
            dict(
                seconds=time.perf_counter() - started,
                sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                note="One shared primitive profile, charged once before all seed selections",
            ),
        )
    for phase in ("selection", "oracle"):
        for seed in args.seeds:
            output = root / str(seed)
            command = [
                sys.executable,
                "-m",
                "experiments.common.comparison",
                "--split-manifest",
                str(profiled_manifest),
                "--output",
                str(output),
                "--oracle-root",
                str(root / "oracle"),
                "--methods",
                *methods,
                "--seed",
                str(seed),
                "--input-seed",
                "123",
                "--phase",
                phase,
                "--top-k",
                "20",
                "--budget-fraction",
                "1",
                "--workers",
                str(args.workers),
                "--warmup",
                "5",
                "--rep",
                "20",
                "--case-timeout",
                "14400",
                "--validation-repeats",
                "7",
                "--wait-idle",
            ]
            if output.exists():
                command.append("--resume")
            with (root / f"{seed}-{phase}.log").open("a") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if phase == "selection":
            frozen = {}
            for seed in args.seeds:
                for device in manifest["devices"]:
                    for workload in manifest["splits"]["test"]:
                        path = root / str(seed) / device["name"] / "test" / workload["name"] / "methods.json"
                        if not path.exists():
                            raise ValueError(f"missing completed selection table: {path}")
                for path in (root / str(seed)).glob("*/test/*/methods.json"):
                    # Every available method must have finished its fixed online
                    # attempt budget before any exhaustive timings are read.
                    values = json.loads(path.read_text())
                    if set(methods) - values.keys():
                        raise ValueError(f"missing selection methods: {path}")
                    rankings = {}
                    for method in methods:
                        report_name = "tiletune" if method.startswith("tiletune") else "xgboost" if method.startswith("xgboost") else method
                        report_path = path.parent / method / (report_name + ".json")
                        if report_path.exists():
                            report = json.loads(report_path.read_text())
                            rankings[method] = dict(ranking=report["ranking"], selection=report["selection"])
                        elif values[method]["status"] != "unavailable":
                            raise ValueError(f"missing frozen ranking: {report_path}")
                    frozen[str(path.relative_to(root))] = dict(methods=values, rankings=rankings)
            snapshot = dict(plan=plan, methods=frozen, sha256=hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest())
            frozen_path = root / "all-selections-frozen.json"
            if frozen_path.exists() and json.loads(frozen_path.read_text()) != snapshot:
                raise ValueError("study selections changed after freezing")
            write_json(frozen_path, snapshot)
    results = []
    for seed in args.seeds:
        comparison = json.loads((root / str(seed) / "comparison.json").read_text())
        results.extend(dict(seed=seed, **item) for item in comparison["results"])
    quality = {}
    for method in methods:
        values = [
            (r["workload"]["op"], r["validation"][method]["performance_vs_brute_force"])
            for r in results
            if r.get("validation") and method in r["validation"]
        ]
        quality[method] = dict(
            median=statistics.median(v for _, v in values) if values else None,
            worst=min((v for _, v in values), default=None),
            families={op: statistics.median(v for family, v in values if family == op) for op in sorted({op for op, _ in values})},
        )
    preparation = dict(primitive_profile=json.loads((root / "profile-preparation.json").read_text()), seeds={})
    for seed in args.seeds:
        costs = {}
        for method in ("xgboost", "xgboost_stratified"):
            collections = [
                json.loads(path.read_text())
                for split in ("train", "validation")
                for path in (root / str(seed)).glob(f"*/{method}/{split}/*/result.json")
            ]
            models = [json.loads(path.read_text()) for path in (root / str(seed)).glob(f"*/models/{method}-*.json")]
            costs[method] = dict(
                collection_worker_seconds=sum(r.get("worker_wall_seconds", 0) for r in collections),
                collection_tuning_seconds=sum(r.get("tuning_seconds", 0) for r in collections),
                fit_seconds=sum(m["training"]["fit_seconds"] for m in models),
            )
        preparation["seeds"][str(seed)] = costs
    write_json(root / "study.json", dict(plan=plan, preparation=preparation, oracle_at_20=quality, results=results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

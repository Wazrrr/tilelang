"""Checkpointed correctness/compilation census of a declared common pool.

python -m experiments.common.census --device ampere --config-space expanded --workloads gemm_square --output results/census
Each shard has an isolated accelerator process. This is development/oracle
collection, not uncharged preparation for an online tuner.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from .execution import compilation_census
from .run import make_request, run_case
from experiments.utils.io import write_json
from .spec import PRESETS
from .spaces import space_summary
from .spec import Device, TARGETS, configuration_space, default_workloads, load_manifest


def main(argv=None, *, family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=TARGETS, default="ampere")
    parser.add_argument("--manifest", type=Path)
    if family:
        parser.add_argument("--suite", choices=("development", "final"), default="development")
    parser.add_argument("--config-space", choices=PRESETS, help="Preset override; defaults to expanded without a manifest")
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--config-indices", nargs="+", type=int)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path, required=family is None, default=Path(f"experiments/{family}/results/census") if family else None
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--wait-idle", action="store_true")
    args = parser.parse_args(argv)
    if any(getattr(args, key) <= 0 for key in ("shard_size", "workers", "warmup", "rep", "timeout", "case_timeout")) or args.seed < 0:
        parser.error("budgets must be positive and seed must be nonnegative")
    devices, workloads = (
        load_manifest(json.loads(args.manifest.read_text()))
        if args.manifest
        else ([Device(args.device, TARGETS[args.device])], default_workloads() if family is None else [])
    )
    if family:
        from experiments.families import FAMILIES
        from experiments.suite import core_cases

        workloads = [w for w in workloads if FAMILIES[w.op] == family] if args.manifest else core_cases(args.suite, [family])
        if not workloads:
            parser.error(f"manifest has no {family} workloads")
        if any(
            d.target["kind"] not in ("cuda", "hip") and any(not w.configs and w.name not in (d.configs or {}) for w in workloads)
            for d in devices
        ):
            parser.error("native configuration grids must be supplied for this backend")
    if args.workloads:
        if set(args.workloads) - {w.name for w in workloads}:
            parser.error("unknown workload name")
        workloads = [w for w in workloads if w.name in args.workloads]
    if args.config_space or not args.manifest:
        workloads = [replace(w, config_space=args.config_space or "expanded") for w in workloads]
    settings = dict(
        method="brute_force",
        metric="pipeline_time",
        top_k=1,
        memory_regime="streaming",
        trace=False,
        **{key: getattr(args, key) for key in ("workers", "warmup", "rep", "timeout", "case_timeout", "seed")},
    )
    cases = []
    for d in devices:
        for w in workloads:
            space = configuration_space(w, d)
            indices = args.config_indices if args.config_indices is not None else list(range(len(space["configs"])))
            if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(space["configs"]) for i in indices):
                parser.error("config indices must be unique and inside every selected pool")
            cases.append(
                dict(workload=w.to_dict(), device=d.to_dict(), space=space_summary(space), config_ids=space["config_ids"], indices=indices)
            )
    plan = dict(version=1, cases=cases, settings=settings, shard_size=args.shard_size)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if (root / "plan.json").exists() and json.loads((root / "plan.json").read_text()) != plan:
        raise ValueError("census plan changed; use a new output directory")
    write_json(root / "plan.json", plan)
    from experiments.utils.cli import source_hashes
    from tilelang.cache.kernel_cache import KernelCache
    from .comparison import wait_for_idle
    from .spec import Workload

    provenance = dict(
        source_sha256=source_hashes("experiments/common/kernels.py", "examples/flash_attention/example_mha_fwd_bshd.py"),
        native_build=KernelCache._get_tilelang_lib_stamp(),
    )
    if (root / "provenance.json").exists() and json.loads((root / "provenance.json").read_text()) != provenance:
        raise ValueError("census source/compiler changed; use a new output directory")
    write_json(root / "provenance.json", provenance)
    summaries = []
    for case in cases:
        w, d = Workload(**case["workload"]), Device(**case["device"])
        directory = root / d.name / w.name
        directory.mkdir(parents=True, exist_ok=True)
        space = configuration_space(w, d)
        write_json(directory / "config-space.json", space)
        records, durations = [], []
        for offset in range(0, len(case["indices"]), args.shard_size):
            indices = case["indices"][offset : offset + args.shard_size]
            shard = directory / f"shard-{offset // args.shard_size:05d}"
            request = make_request(w, d, dict(settings, config_indices=indices))
            if (shard / "result.json").exists():
                if json.loads((shard / "request.json").read_text()) != request:
                    raise ValueError("census shard request changed")
                result = json.loads((shard / "result.json").read_text())
            else:
                if shard.exists():
                    shard.rename(shard.with_name(shard.name + f".interrupted-{time.time_ns()}"))
                wait_for_idle(root, wait=args.wait_idle, allow_contended=False)
                result = run_case(request, shard)
            if (shard / "outcomes.json").exists():
                for record in json.loads((shard / "outcomes.json").read_text()):
                    record["shard_index"] = record["index"]
                    record["index"] = record["original_index"]
                    records.append(record)
            else:
                records.extend(
                    dict(
                        index=i,
                        original_index=i,
                        config=space["configs"][i],
                        config_id=space["config_ids"][i],
                        status="shard_failed",
                        error=result.get("reason"),
                    )
                    for i in indices
                )
            durations.append(result.get("tuning_seconds"))
            correct = [r for r in records if r["status"] == "benchmarked"]
            old = [r for r in correct if r["original_index"] < space["retained_current_count"]]
            summary = dict(
                device=d.name,
                workload=w.name,
                space=space_summary(space),
                **compilation_census(records),
                requested_count=len(case["indices"]),
                completed_count=len(records),
                best_ms=min((r["latency_ms"] for r in correct), default=None),
                current_best_ms=min((r["latency_ms"] for r in old), default=None),
                tuning_seconds=sum(durations) if all(t is not None for t in durations) else None,
            )
            write_json(directory / "outcomes.json", records)
            write_json(directory / "census.json", summary)
            print(f"{d.name}/{w.name}: {len(records)}/{len(case['indices'])}, {len(correct)} correct", flush=True)
        summaries.append(summary)
        write_json(root / "summary.json", summaries)
    return int(any(s["correct_count"] == 0 or s["statuses"].get("shard_failed", 0) for s in summaries))


if __name__ == "__main__":
    raise SystemExit(main())

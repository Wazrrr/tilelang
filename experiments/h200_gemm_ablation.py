"""Run the fixed 4096^3 BF16 GEMM system/TileTune ablation on H200.

The six modes form one cumulative experiment:

    bruteforce -> grouped -> grouped_pipeline -> combined -> combined_post -> combined_tiletune

All modes use the same ordered 576-configuration pool and one total pool of
64 compiler workers.  ``grouped`` adds grouped compilation,
``grouped_pipeline`` adds compilation/benchmark overlap, and ``combined`` adds
four-GPU benchmarking;
``combined_post`` adds only the H200 post-compile policy, with no pre-lowering
analysis, ranking, or selection. ``combined_tiletune`` then adds strict memory
ranking at alpha=0.5. Grouped modes share one host executable per
compile group but defer its setup until first benchmark use, keeping runtime
setup out of the compilation phase for a clean compile/benchmark ablation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.common import h200
from experiments.families import family_module
from experiments.utils.io import write_json


ABLATION_SETTINGS = {
    **h200.EXPERIMENT_SETTINGS,
    "post_compile_only": dict(ranking_metric=None, alpha=None, post_compile_filter=True),
}

MODE_SPECS = {
    "bruteforce": dict(variant="baseline", gpu_count=1, settings="E1"),
    "grouped": dict(variant="grouped", gpu_count=1, settings="E1"),
    "grouped_pipeline": dict(variant="pipeline_grouped", gpu_count=1, settings="E1"),
    "combined": dict(variant="combined", gpu_count=4, settings="E2"),
    "combined_post": dict(variant="combined", gpu_count=4, settings="post_compile_only"),
    "combined_tiletune": dict(variant="tiletune", gpu_count=4, settings="E3"),
}
WORKLOAD_NAME = "gemm_square"


def study_plan(modes=None):
    """Return the frozen six-mode plan without importing TileLang or CUDA."""
    selected = set(MODE_SPECS if modes is None else modes)
    if not selected or selected - MODE_SPECS.keys():
        raise ValueError("modes must be distinct supported ablation mode names")
    workloads = family_module("gemm", "cases").cases(holdout=True)
    workload = next(workload for workload in workloads if workload.name == WORKLOAD_NAME)
    configs = family_module("gemm", "spaces").get_configs()
    indices = list(range(len(configs)))
    plan = []
    for mode, spec in MODE_SPECS.items():
        if mode not in selected:
            continue
        plan.append(
            dict(
                experiment=mode,
                variant=spec["variant"],
                workload=workload.to_dict(),
                indices=indices,
                configs=configs,
                gpu_count=spec["gpu_count"],
                settings=dict(
                    h200.SETTINGS,
                    **ABLATION_SETTINGS[spec["settings"]],
                    grouped_compile_runtime_setup="lazy",
                    preflight=False,
                ),
            )
        )
    return plan


def experiment_identity():
    """Freeze every measured source, native library, config, and setting."""
    return dict(
        version=1,
        plan=study_plan(),
        source_identity=h200.source_identity(),
        python=str(Path(sys.executable).resolve()),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )


def aggregate_completed(root):
    """Publish comparable E2E and marginal speedups from completed modes."""
    rows = []
    for item in study_plan():
        case_dir = root / item["experiment"] / WORKLOAD_NAME
        marker = case_dir / "completed.json"
        if not marker.exists():
            continue
        completion = h200.read(marker)
        output = case_dir / completion["attempt"]
        summary = h200.read(output / "summary.json")
        monitor = h200.read(output / "monitor.json")
        rows.append(
            dict(
                mode=item["experiment"],
                variant=item["variant"],
                attempt=str(output.relative_to(root)),
                gpu_count=item["gpu_count"],
                config_count=summary["config_count"],
                candidate_statuses=summary["candidate_statuses"],
                tuning_seconds=summary["tuning_seconds"],
                end_to_end_seconds=monitor["wall_seconds"],
                gpu_seconds=monitor["wall_seconds"] * item["gpu_count"],
                winner_latency_ms=summary["winner_latency_ms"],
                winner_config=summary["winner_config"],
            )
        )
    if rows:
        baseline = rows[0]["end_to_end_seconds"] if rows[0]["mode"] == "bruteforce" else None
        previous = None
        for row in rows:
            duration = row["end_to_end_seconds"]
            row["normalized_end_to_end"] = duration / baseline if baseline is not None else None
            row["cumulative_speedup_vs_bruteforce"] = baseline / duration if baseline is not None else None
            row["incremental_speedup"] = previous / duration if previous is not None else 1.0
            previous = duration
    write_json(root / "ablation-comparison.json", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--modes", choices=list(MODE_SPECS), nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    requested_modes = list(MODE_SPECS) if args.modes is None else args.modes
    if len(requested_modes) != len(set(requested_modes)):
        parser.error("modes must be distinct")
    requested_plan = study_plan(requested_modes)
    identity = experiment_identity()
    if args.plan:
        print(json.dumps(dict(identity, requested_modes=requested_modes), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required")

    root = args.output.resolve()
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise ValueError("resume requires an existing ablation manifest")
        manifest = h200.read(manifest_path)
        frozen = {key: manifest[key] for key in identity}
        if frozen != identity:
            raise ValueError("ablation identity changed: plan, measured code, interpreter, revision, or native build differ")
    else:
        if root.exists():
            raise FileExistsError("output already exists; use --resume to verify and continue it")
        manifest = dict(identity, allocations=[])

    from experiments.utils.isolation import interruptible, measurement_lease, select_cpu_pools
    from experiments.utils.monitor import snapshot

    required_gpu_count = max(item["gpu_count"] for item in requested_plan)
    gpus = h200.select_gpus(snapshot(), args.gpus, count=required_gpu_count)
    affinity = set(os.sched_getaffinity(0))
    cpu_pools = select_cpu_pools(h200.SETTINGS["compiler_workers_total"], 1)
    assigned = set(cpu_pools[0])
    spare = affinity - assigned
    if not spare:
        raise RuntimeError("no CPU remains for the experiment coordinator")

    allocation = dict(
        modes=requested_modes,
        gpus=gpus,
        cpu_pools=cpu_pools,
    )
    if not manifest.get("allocations") or manifest["allocations"][-1] != allocation:
        manifest.setdefault("allocations", []).append(allocation)
    root.mkdir(parents=True, exist_ok=args.resume)
    write_json(manifest_path, manifest)

    with measurement_lease(gpus) as leases, interruptible():
        os.sched_setaffinity(0, spare)
        try:
            h200.run_queue(
                root,
                requested_plan,
                gpus,
                cpu_pools,
                leases,
                code_identity=identity["source_identity"],
            )
        finally:
            os.sched_setaffinity(0, affinity)
            aggregate_completed(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check fresh CPU IR analysis against a frozen memory replay, then join oracles.

The archived experiment builders only elaborate the original programs. The live
analyzer sees PrimFuncs and device inputs; family/timing dispatch is disabled.
No compilation, GPU execution, or oracle lookup occurs in the analysis workers.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from experiments.compare_results import compare
from experiments.utils.results import provenance, read


def frozen_builders(study):
    """Keep the live analyzer; import experiment/example builders from the archive."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tilelang.tiletune.engine as engine
    import tilelang.tiletune.families.base as families

    def forbidden(*args, **kwargs):
        raise AssertionError("memory analysis invoked a family or timing helper")

    engine.select_specialization = forbidden
    engine.predict_warp_specialization = forbidden
    engine.pipeline.analyze_pipeline = forbidden
    engine.occupancy.analyze_waves = forbidden
    families.KernelSpecialization.__init__ = forbidden
    sources = Path(study) / "sources"
    # Each worker has its own module namespace. No shared environment is changed.
    for name in list(sys.modules):
        if name in ("examples", "experiments") or name.startswith(("examples.", "experiments.")):
            del sys.modules[name]
    sys.path.insert(0, str(sources))


def validate_case(study, replay_row, output, alpha):
    from tilelang.tiletune import TileTuneConfig, analyze_prim_func
    from tiletune_core.ranking import alpha_budget, rank_records, select_top_k
    from experiments.common.kernels import make_case
    from experiments.common.spec import Workload

    study, output = Path(study), Path(output)
    name, family = replay_row["workload"], replay_row["family"]
    source = replay_row["source"]
    raw = Path(source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"archived analysis changed: {name}")
    original = json.loads(raw)
    del raw
    replay_ref = replay_row["ranking"]
    raw = Path(replay_ref["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != replay_ref["sha256"]:
        raise ValueError(f"replay ranking changed: {name}")
    expected = json.loads(raw)
    del raw
    case_info = read(study / "comparison" / family / name / "comparison.json")
    workload = Workload(**case_info["workload"])
    case = make_case(workload)
    targets = {device["name"]: device["target"] for device in read(study / "device-manifest.json")}
    target = targets[case_info["device"]]
    settings = {key: value for key, value in original["settings"].items() if key in {field.name for field in fields(TileTuneConfig)}}
    settings.update(
        enabled=True,
        ranking_metric="memory",
        performance_model=None,
        top_k=None,
        alpha=alpha,
        report_path=None,
        trace_path=None,
        facts_path=None,
    )
    config = TileTuneConfig(**settings)
    del original
    records, mismatches = [], []
    started = time.perf_counter()
    for saved in expected["configs"]:
        index = saved["index"]
        func = case.build(**saved["config"])
        before = func.script()
        result = analyze_prim_func(func, config, target=target, pass_configs=case.pass_configs)
        if func.script() != before:
            raise AssertionError(f"analysis changed PrimFunc {name}/{index}")
        cost = result["tile_cost"]
        # Check interpretable components as well as the exact composite integer.
        keys = ("score", "logical_byte_waves", "logical_memory_access_waves", "pipeline_depth")
        differences = {
            key: {"live": cost.get(key), "replay": saved["tile_cost"].get(key)}
            for key in keys
            if cost.get(key) != saved["tile_cost"].get(key)
        }
        if differences:
            mismatches.append({"index": index, "differences": differences})
        records.append(
            dict(index=index, config=saved["config"], status="analyzed", tile_cost=cost, pre_lowering=result["pressure"]["decision"])
        )
    ranking = rank_records(records)
    budget = alpha_budget(len(records), alpha)
    selected = select_top_k(ranking, budget, strict_budget=True)
    elapsed = time.perf_counter() - started
    folder = output / family / name
    folder.mkdir(parents=True, exist_ok=False)
    ranked_path = folder / "tiletune.json"
    report = dict(
        version=2,
        settings=config.to_cache_key_dict(),
        configs=records,
        ranking=ranking,
        selection=dict(
            alpha=alpha,
            pool_size=len(records),
            requested_k=budget,
            selected_indices=selected,
            selected_count=len(selected),
            strict_budget=True,
            budget_excess=0,
        ),
        source=source,
        replay=replay_ref,
        analysis_seconds=elapsed,
        mismatches=mismatches,
        measurement_scope="fresh CPU IR analysis; no compilation or GPU execution",
    )
    ranked_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = Path(source["path"]).parent / "experiment.json"
    if manifest.is_file():
        (folder / "experiment.json").write_bytes(manifest.read_bytes())
    return dict(
        workload=name,
        family=family,
        pool_size=len(records),
        scored_count=sum(r["tile_cost"]["score"] is not None for r in records),
        selected_count=len(selected),
        alpha_budget=budget,
        analysis_seconds=elapsed,
        mismatches=mismatches,
        ranking=provenance(ranked_path),
    )


def validate(study, replay, output, alpha, workers):
    study, replay, output = Path(study).resolve(), Path(replay).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("validation output already exists; use a new directory")
    hashes = read(study / "source-hashes.json")
    for relative, expected_hash in hashes.items():
        path = study / "sources" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"archived source changed: {relative}")
    summary = read(replay / "summary.json")
    if summary["study"] != provenance(study / "comparison.json"):
        raise ValueError("replay belongs to a different or modified study")
    output.mkdir(parents=True)
    rows = []
    # Fork before importing TileLang in the parent; workers elaborate independently.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork"), initializer=frozen_builders, initargs=(str(study),)
    ) as executor:
        tasks = [executor.submit(validate_case, str(study), row, str(output), alpha) for row in summary["rows"]]
        for future in as_completed(tasks):
            row = future.result()
            # The complete live ranking and selection are already on disk.
            saved = read(study / "comparison" / row["family"] / row["workload"] / "oracle-curves.json")
            comparison = compare(saved["oracle"]["sources"][0]["path"], {"tiletune": row["ranking"]["path"]}, [row["alpha_budget"]])
            if comparison["oracle"]["sources"] != saved["oracle"]["sources"]:
                raise ValueError("oracle changed since the frozen study")
            method = comparison["methods"][0]
            live_report = read(Path(row["ranking"]["path"]))
            by_index = {record["index"]: record["tile_cost"] for record in live_report["configs"]}
            for candidate in method["oracle_candidates"]:
                if candidate["index"] is not None:
                    cost = by_index[candidate["index"]]
                    candidate.update(
                        {key: cost[key] for key in ("score", "logical_byte_waves", "logical_memory_access_waves", "pipeline_depth")}
                    )
            rank = method["first_oracle_hit_k"]
            row.update(
                first_oracle_hit_k=rank,
                oracle_rank_fraction=rank / row["pool_size"] if rank is not None else None,
                hits_alpha=rank is not None and rank <= row["alpha_budget"],
                oracle_candidates=method["oracle_candidates"],
            )
            curves_path = Path(row["ranking"]["path"]).parent / "oracle-curves.json"
            curves_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
            rows.append(row)
            print(f"{row['workload']}: live oracle tail {rank}/{row['pool_size']}; replay mismatches {len(row['mismatches'])}", flush=True)
            (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    rows.sort(key=lambda row: row["workload"])
    result = dict(
        study=provenance(study / "comparison.json"),
        replay=provenance(replay / "summary.json"),
        source_hashes=provenance(study / "source-hashes.json"),
        rows=rows,
        alpha=alpha,
        analyzed_count=sum(row["pool_size"] for row in rows),
        scored_count=sum(row["scored_count"] for row in rows),
        hits_alpha=sum(row["hits_alpha"] for row in rows),
        replay_mismatches=sum(len(row["mismatches"]) for row in rows),
        family_helpers_disabled=True,
        primfuncs_unchanged=True,
        gpu_work=False,
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = validate(args.study, args.replay, args.output, args.alpha, args.workers)
    print(f"Fresh IR: {result['hits_alpha']}/{len(result['rows'])} at alpha={args.alpha}; {result['replay_mismatches']} replay mismatches")
    return 0 if result["hits_alpha"] == len(result["rows"]) and result["replay_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

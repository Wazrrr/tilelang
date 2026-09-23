"""Fit one CPU XGBoost model per completed exhaustive event workload.

This is a retrospective, within-workload study.  Configuration hashes choose
disjoint training, validation, and held-out sets before outcome status or
latency is inspected.  Failed selected configurations consume the label budget
and are not replaced.

The source event runner records one-GPU wall time only for the complete oracle,
not for arbitrary subsets.  Counterfactual label-collection time is therefore
estimated in direct proportion to the number of attempted configurations.  The
report keeps that estimate separate from the lower-bound CUDA benchmark budget
and from measured CPU fitting time.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from .data import canonical_workload, digest, features
from .model import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_ROUNDS,
    DEFAULT_SUBSAMPLE,
    EARLY_STOPPING_ROUNDS,
    _libraries,
    _matrix,
    _schema,
)


VERSION = 1
SPLIT_POLICY = "seeded_disjoint_config_hash_v1"


def _read_json(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _event_context(experiment):
    """Translate a B200 event record into the feature context used by XGBoost."""
    target = experiment["target"]
    devices = experiment["devices"]
    if len(devices) != 1:
        raise ValueError("within-workload training requires a one-GPU oracle")
    arch = target.get("arch")
    if not arch:
        raise ValueError("event oracle is missing target architecture")
    if target["kind"] == "cuda":
        arch = arch.rstrip("af")
    source_identity = experiment.get("source_identity")
    if not source_identity:
        raise ValueError("event oracle is missing source identity")
    return dict(
        workload=canonical_workload(experiment["workload"]),
        implementation="event." + experiment["workload"]["op"],
        device=dict(kind=target["kind"], arch=arch, name=devices[0]["name"]),
        benchmark_backend=experiment.get("measurement", {}).get("backend", "event"),
        kernel_sha256=dict(source_identity=digest(source_identity)),
        environment=experiment.get("environment"),
    )


def split_indices(workload, configs, *, training_fraction=0.1, validation_fraction=0.1, seed=123):
    """Choose disjoint attempted subsets without reading statuses or labels."""
    for name, value in (("training_fraction", training_fraction), ("validation_fraction", validation_fraction)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if training_fraction + validation_fraction >= 1:
        raise ValueError("training and validation fractions must leave a held-out set")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(configs) < 4 or len({digest(config) for config in configs}) != len(configs):
        raise ValueError("at least four unique configurations are required")
    order = sorted(
        range(len(configs)),
        key=lambda index: digest(dict(policy=SPLIT_POLICY, seed=seed, workload=workload, config=configs[index])),
    )
    training_count = max(2, math.ceil(len(configs) * training_fraction))
    validation_count = max(1, math.ceil(len(configs) * validation_fraction))
    if training_count + validation_count >= len(configs):
        raise ValueError("split fractions leave no held-out configurations")
    return dict(
        policy=SPLIT_POLICY,
        seed=seed,
        training_fraction=training_fraction,
        validation_fraction=validation_fraction,
        training=order[:training_count],
        validation=order[training_count : training_count + validation_count],
        heldout=order[training_count + validation_count :],
    )


def read_event_oracles(root):
    """Read completed workload directories produced by common.b200 E1."""
    root = Path(root).resolve()
    completed_paths = sorted(root.glob("*/completed.json"))
    if not completed_paths:
        raise ValueError(f"no immediate completed workloads under {root}")
    oracles = []
    for completed_path in completed_paths:
        completed = _read_json(completed_path)
        attempt = completed_path.parent / completed["attempt"]
        required = ("experiment.json", "outcomes.json", "summary.json")
        for name in required:
            path = attempt / name
            expected = completed.get("files", {}).get(name)
            if not path.exists() or expected is None or _sha256(path) != expected:
                raise ValueError(f"{completed_path}: missing or changed {name}")
        experiment = _read_json(attempt / "experiment.json")
        outcomes = _read_json(attempt / "outcomes.json")
        summary = _read_json(attempt / "summary.json")
        configs = experiment.get("configs")
        if (
            experiment.get("variant") != "baseline"
            or summary.get("status") != "completed"
            or not isinstance(configs, list)
            or len(configs) != len(outcomes)
            or summary.get("config_count") != len(configs)
        ):
            raise ValueError(f"{attempt}: expected a complete E1 baseline oracle")
        by_index = {record.get("index"): record for record in outcomes}
        if set(by_index) != set(range(len(configs))):
            raise ValueError(f"{attempt}: incomplete or duplicate candidate indices")
        for index, config in enumerate(configs):
            if by_index[index].get("config") != config:
                raise ValueError(f"{attempt}: configuration mismatch at index {index}")
        tuning_seconds = summary.get("tuning_seconds")
        if not isinstance(tuning_seconds, int | float) or not math.isfinite(tuning_seconds) or tuning_seconds <= 0:
            raise ValueError(f"{attempt}: invalid tuning_seconds")
        oracles.append(
            dict(
                name=experiment["workload"]["name"],
                attempt=attempt,
                experiment=experiment,
                configs=configs,
                outcomes=by_index,
                summary=summary,
                context=_event_context(experiment),
                provenance=dict(
                    completed=str(completed_path),
                    attempt=str(attempt),
                    experiment_sha256=_sha256(attempt / "experiment.json"),
                    outcomes_sha256=_sha256(attempt / "outcomes.json"),
                    summary_sha256=_sha256(attempt / "summary.json"),
                ),
            )
        )
    return oracles


def _successful_rows(oracle, indices):
    rows, labels, successful_indices = [], [], []
    for index in indices:
        outcome = oracle["outcomes"][index]
        if outcome.get("status") != "benchmarked":
            continue
        latency = outcome.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, int | float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"{oracle['attempt']}: invalid latency at index {index}")
        rows.append(features(oracle["context"], oracle["configs"][index]))
        labels.append(math.log(latency))
        successful_indices.append(index)
    return rows, labels, successful_indices


def _rmse(np, predictions, labels):
    return float(np.sqrt(np.mean((predictions - np.asarray(labels)) ** 2)))


def train_oracle(
    oracle,
    output,
    *,
    training_fraction=0.1,
    validation_fraction=0.1,
    rounds=DEFAULT_ROUNDS,
    max_depth=DEFAULT_MAX_DEPTH,
    learning_rate=DEFAULT_LEARNING_RATE,
    subsample=DEFAULT_SUBSAMPLE,
    seed=123,
    workers=8,
    top_k=20,
):
    """Train and serialize one model, returning its complete timing report."""
    np, xgb = _libraries()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if any(type(value) is not int or value <= 0 for value in (rounds, max_depth, workers, top_k)):
        raise ValueError("rounds, depth, workers, and top_k must be positive integers")
    if not 0 < learning_rate <= 1 or not 0 < subsample <= 1:
        raise ValueError("learning rate and subsample must be in (0, 1]")

    configs = oracle["configs"]
    split = split_indices(
        oracle["context"]["workload"],
        configs,
        training_fraction=training_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    preparation_started = time.perf_counter()
    # The full declared pool is known before measurement.  It supplies schema
    # categories, never labels, matching the ordinary baseline contract.
    all_rows = [features(oracle["context"], config) for config in configs]
    schema = _schema(all_rows)
    train_rows, train_labels, train_successes = _successful_rows(oracle, split["training"])
    valid_rows, valid_labels, valid_successes = _successful_rows(oracle, split["validation"])
    heldout_rows, heldout_labels, heldout_successes = _successful_rows(oracle, split["heldout"])
    if len(train_rows) < 2 or not valid_rows or not heldout_rows:
        raise ValueError(f"{oracle['name']}: insufficient successful labels in a frozen split")
    feature_preparation_seconds = time.perf_counter() - preparation_started

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cpu",
        max_depth=max_depth,
        eta=learning_rate,
        subsample=subsample,
        seed=seed,
        nthread=workers,
    )
    fit_started = time.perf_counter()
    train_data = xgb.DMatrix(_matrix(train_rows, schema), label=train_labels, nthread=workers)
    valid_data = xgb.DMatrix(_matrix(valid_rows, schema), label=valid_labels, nthread=workers)
    booster = xgb.train(
        params,
        train_data,
        num_boost_round=rounds,
        evals=[(valid_data, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_rounds = int(booster.best_iteration) + 1
    validation_predictions = booster.predict(valid_data, iteration_range=(0, best_rounds))
    fit_seconds = time.perf_counter() - fit_started

    evaluation_started = time.perf_counter()
    heldout_all_rows = [all_rows[index] for index in split["heldout"]]
    heldout_data = xgb.DMatrix(_matrix(heldout_all_rows, schema), nthread=workers)
    heldout_predictions = booster.predict(heldout_data, iteration_range=(0, best_rounds))
    heldout_positions = {index: position for position, index in enumerate(split["heldout"])}
    heldout_success_predictions = np.asarray([heldout_predictions[heldout_positions[index]] for index in heldout_successes])
    ranked = sorted(
        range(len(split["heldout"])),
        key=lambda position: (heldout_predictions[position], split["heldout"][position]),
    )
    selected = ranked[: min(top_k, len(ranked))]
    selected_indices = [split["heldout"][position] for position in selected]
    heldout_latencies = [math.exp(value) for value in heldout_labels]
    oracle_best = min(heldout_latencies)
    selected_latencies = [
        oracle["outcomes"][index]["latency_ms"] for index in selected_indices if oracle["outcomes"][index].get("status") == "benchmarked"
    ]
    selected_best = min(selected_latencies) if selected_latencies else None
    evaluation_seconds = time.perf_counter() - evaluation_started

    pool_size = len(configs)
    full_collection_seconds = oracle["summary"]["tuning_seconds"]
    training_collection_seconds = full_collection_seconds * len(split["training"]) / pool_size
    validation_collection_seconds = full_collection_seconds * len(split["validation"]) / pool_size
    measurement = oracle["experiment"].get("measurement", {})
    benchmark_budget_ms = measurement.get("warmup_ms", 0) + measurement.get("rep_ms", 0)
    if not isinstance(benchmark_budget_ms, int | float) or benchmark_budget_ms < 0:
        benchmark_budget_ms = 0
    benchmark_lower_bound_seconds = benchmark_budget_ms * (len(train_successes) + len(valid_successes)) / 1000

    statuses = Counter(record.get("status") for record in oracle["outcomes"].values())
    report = dict(
        workload=oracle["name"],
        workload_description=oracle["experiment"]["workload"],
        pool_size=pool_size,
        outcome_statuses=dict(statuses),
        split=dict(
            policy=SPLIT_POLICY,
            seed=seed,
            training_fraction=training_fraction,
            validation_fraction=validation_fraction,
            training_attempted=len(split["training"]),
            training_successful=len(train_successes),
            validation_attempted=len(split["validation"]),
            validation_successful=len(valid_successes),
            heldout_attempted=len(split["heldout"]),
            heldout_successful=len(heldout_successes),
            training_indices=sorted(split["training"]),
            validation_indices=sorted(split["validation"]),
            heldout_indices=sorted(split["heldout"]),
            failure_policy="selected failures consume the attempted label budget and are not replaced",
        ),
        cpu_timing=dict(
            feature_preparation_seconds=feature_preparation_seconds,
            fit_seconds=fit_seconds,
            heldout_evaluation_seconds=evaluation_seconds,
        ),
        label_collection_estimate=dict(
            source_full_oracle_tuning_seconds=full_collection_seconds,
            source_full_oracle_gpu_count=oracle["summary"].get("benchmark_gpu_count"),
            training_collection_seconds=training_collection_seconds,
            validation_collection_seconds=validation_collection_seconds,
            total_collection_seconds=training_collection_seconds + validation_collection_seconds,
            cuda_benchmark_budget_lower_bound_seconds=benchmark_lower_bound_seconds,
            method="source one-GPU tuning_seconds multiplied by attempted subset size / full pool size",
            scope=("compilation pipeline, correctness, and CUDA-event measurement; counterfactual estimate, not a subset rerun"),
        ),
        boosting=dict(
            params=params,
            max_rounds=rounds,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            trained_rounds=booster.num_boosted_rounds(),
            best_rounds=best_rounds,
        ),
        quality=dict(
            validation_log_rmse=_rmse(np, validation_predictions, valid_labels),
            heldout_log_rmse=_rmse(np, heldout_success_predictions, heldout_labels),
            heldout_top_k=len(selected),
            heldout_selected_successful=len(selected_latencies),
            heldout_oracle_best_latency_ms=oracle_best,
            heldout_selected_best_latency_ms=selected_best,
            heldout_oracle_at_k=oracle_best / selected_best if selected_best is not None else None,
            heldout_selected_indices=selected_indices,
        ),
        provenance=oracle["provenance"],
    )
    serialization_started = time.perf_counter()
    booster_payload = json.loads(booster.save_raw(raw_format="json"))
    serialization_seconds = time.perf_counter() - serialization_started
    report["cpu_timing"]["serialization_seconds"] = serialization_seconds
    report["cpu_timing"]["training_pipeline_seconds"] = feature_preparation_seconds + fit_seconds + serialization_seconds
    report["estimated_training_total_seconds"] = (
        report["label_collection_estimate"]["total_collection_seconds"] + report["cpu_timing"]["training_pipeline_seconds"]
    )
    artifact = dict(
        version=VERSION,
        method="xgboost_within_workload_config_features",
        label="log(latency_ms)",
        xgboost_version=xgb.__version__,
        schema=schema,
        report=report,
        booster=booster_payload,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def train_all(
    oracle_root,
    output,
    *,
    training_fraction=0.1,
    validation_fraction=0.1,
    rounds=DEFAULT_ROUNDS,
    max_depth=DEFAULT_MAX_DEPTH,
    learning_rate=DEFAULT_LEARNING_RATE,
    subsample=DEFAULT_SUBSAMPLE,
    seed=123,
    workers=8,
    top_k=20,
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    reports = []
    for oracle in read_event_oracles(oracle_root):
        report = train_oracle(
            oracle,
            output / "models" / f"{oracle['name']}.json",
            training_fraction=training_fraction,
            validation_fraction=validation_fraction,
            rounds=rounds,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            seed=seed,
            workers=workers,
            top_k=top_k,
        )
        reports.append(report)
        oracle_at_k = report["quality"]["heldout_oracle_at_k"]
        quality = "N/A" if oracle_at_k is None else f"{100 * oracle_at_k:.2f}%"
        print(
            f"{report['workload']}: fit={report['cpu_timing']['fit_seconds']:.3f}s "
            f"labels~{report['label_collection_estimate']['total_collection_seconds']:.3f}s "
            f"heldout@{report['quality']['heldout_top_k']}={quality}",
            flush=True,
        )
    aggregate = dict(
        version=VERSION,
        protocol="within-workload retrospective XGBoost with disjoint configuration splits",
        recorded_at=datetime.now(timezone.utc).isoformat(),
        oracle_root=str(Path(oracle_root).resolve()),
        output=str(output),
        settings=dict(
            split_policy=SPLIT_POLICY,
            training_fraction=training_fraction,
            validation_fraction=validation_fraction,
            rounds=rounds,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            seed=seed,
            workers=workers,
            top_k=top_k,
            device="cpu",
        ),
        workloads=reports,
        totals=dict(
            workload_count=len(reports),
            training_attempted=sum(r["split"]["training_attempted"] for r in reports),
            training_successful=sum(r["split"]["training_successful"] for r in reports),
            validation_attempted=sum(r["split"]["validation_attempted"] for r in reports),
            validation_successful=sum(r["split"]["validation_successful"] for r in reports),
            training_collection_estimated_seconds=sum(r["label_collection_estimate"]["training_collection_seconds"] for r in reports),
            validation_collection_estimated_seconds=sum(r["label_collection_estimate"]["validation_collection_seconds"] for r in reports),
            label_collection_estimated_seconds=sum(r["label_collection_estimate"]["total_collection_seconds"] for r in reports),
            cuda_benchmark_budget_lower_bound_seconds=sum(
                r["label_collection_estimate"]["cuda_benchmark_budget_lower_bound_seconds"] for r in reports
            ),
            feature_preparation_seconds=sum(r["cpu_timing"]["feature_preparation_seconds"] for r in reports),
            fit_seconds=sum(r["cpu_timing"]["fit_seconds"] for r in reports),
            serialization_seconds=sum(r["cpu_timing"]["serialization_seconds"] for r in reports),
            cpu_training_pipeline_seconds=sum(r["cpu_timing"]["training_pipeline_seconds"] for r in reports),
            estimated_training_total_seconds=sum(r["estimated_training_total_seconds"] for r in reports),
            process_wall_seconds=time.perf_counter() - started,
        ),
        accounting_note=(
            "The primary estimated training total is counterfactual one-GPU label-collection wall time plus "
            "measured CPU feature preparation, fitting, and serialization. Held-out evaluation time and the "
            "already-paid full oracle sweep are excluded."
        ),
    )
    summary = output / "training-summary.json"
    with summary.open("x") as stream:
        json.dump(aggregate, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-fraction", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    result = train_all(
        args.oracle_root,
        args.output,
        training_fraction=args.training_fraction,
        validation_fraction=args.validation_fraction,
        rounds=args.rounds,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        seed=args.seed,
        workers=args.workers,
        top_k=args.top_k,
    )
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()

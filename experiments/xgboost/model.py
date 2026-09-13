"""Upstream tree boosting on declared configuration features and log latency.

This is a standalone baseline, independent of TileTune's analytical features,
pressure gates and primitive profiles. Training and inference run on the CPU.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .data import digest, domain, features, workload_key

VERSION = 1


def _libraries():
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as error:
        raise ImportError("XGBoost baseline requires: pip install -r experiments/xgboost/requirements.txt") from error
    return np, xgb


def _schema(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    schema = []
    for key in keys:
        values = [row[key] for row in rows if key in row]
        strings = [isinstance(value, str) for value in values]
        if any(strings) and not all(strings):
            raise ValueError(f"feature changes type: {key}")
        schema.append(dict(name=key, categories=sorted(set(values)) if all(strings) else None))
    return schema


def _matrix(rows, schema):
    np, _ = _libraries()
    names = {field["name"] for field in schema}
    width = sum(len(field["categories"]) + 1 if field["categories"] is not None else 1 for field in schema)
    result = np.full((len(rows), width), np.nan, dtype="float32")
    for i, row in enumerate(rows):
        unknown = row.keys() - names
        if unknown:
            raise ValueError(f"features absent from the training schema: {sorted(unknown)}")
        offset = 0
        for field in schema:
            categories = field["categories"]
            value = row.get(field["name"])
            if categories is None:
                if value is not None:
                    if isinstance(value, str):
                        raise ValueError(f"numeric feature changed type: {field['name']}")
                    result[i, offset] = value
                offset += 1
            else:
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"categorical feature changed type: {field['name']}")
                result[i, offset : offset + len(categories) + 1] = 0
                result[i, offset + (categories.index(value) if value in categories else len(categories))] = 1
                offset += len(categories) + 1
    if np.isinf(result).any():
        raise ValueError("feature exceeds float32 range")
    return result


def _samples(runs):
    # Repeated measurements do not give one candidate extra statistical weight.
    repeated = defaultdict(list)
    items = {}
    for run in runs:
        context = run["context"]
        for sample in run["samples"]:
            key = digest(dict(context=context, config=sample["config"]))
            repeated[key].append(sample["latency_ms"])
            items[key] = (context, sample["config"])
    rows, labels, groups = [], [], []
    for key in sorted(items):
        context, config = items[key]
        rows.append(features(context, config))
        labels.append(math.log(statistics.median(repeated[key])))
        groups.append(workload_key(context))
    return rows, labels, groups


def train(training, validation, output, *, rounds=200, max_depth=6, learning_rate=0.05, seed=123, workers=4):
    """Fit on training workloads; early stopping uses separate validation cases."""
    np, xgb = _libraries()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not training or not validation:
        raise ValueError("explicit training and validation runs are required")
    if any(type(value) is not int or value <= 0 for value in (rounds, max_depth, workers)) or not 0 < learning_rate <= 1:
        raise ValueError("invalid training budgets or learning rate")
    train_keys = {workload_key(run["context"]) for run in training}
    validation_keys = {workload_key(run["context"]) for run in validation}
    if train_keys & validation_keys:
        raise ValueError("training/validation workload overlap; split whole shapes, not configurations or run names")
    domains = {digest(domain(run["context"])): domain(run["context"]) for run in training}
    if any(digest(domain(run["context"])) not in domains for run in validation):
        raise ValueError("validation must use a training device, kernel implementation/source, and benchmark backend")
    rows, labels, groups = _samples(training)
    valid_rows, valid_labels, _ = _samples(validation)
    if len(rows) < 2:
        raise ValueError("at least two measured training candidates are required")
    schema = _schema(rows)
    counts = Counter(groups)
    weights = np.asarray([len(rows) / (len(counts) * counts[group]) for group in groups])
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cpu",
        max_depth=max_depth,
        eta=learning_rate,
        seed=seed,
        nthread=workers,
    )
    started = time.perf_counter()
    train_data = xgb.DMatrix(_matrix(rows, schema), label=labels, weight=weights, nthread=workers)
    valid_data = xgb.DMatrix(_matrix(valid_rows, schema), label=valid_labels, nthread=workers)
    booster = xgb.train(
        params, train_data, num_boost_round=rounds, evals=[(valid_data, "validation")], early_stopping_rounds=20, verbose_eval=False
    )
    best_rounds = int(booster.best_iteration) + 1
    prediction = booster.predict(valid_data, iteration_range=(0, best_rounds))
    report = dict(
        training_samples=len(rows),
        validation_samples=len(valid_rows),
        training_workloads=sorted(train_keys),
        validation_workloads=sorted(validation_keys),
        fit_seconds=time.perf_counter() - started,
        validation_log_rmse=float(np.sqrt(np.mean((prediction - np.asarray(valid_labels)) ** 2))),
        training_runs=[run["provenance"] for run in training],
        validation_runs=[run["provenance"] for run in validation],
        training_workload_descriptions={workload_key(run["context"]): run["context"]["workload"] for run in training},
        validation_workload_descriptions={workload_key(run["context"]): run["context"]["workload"] for run in validation},
    )
    for name, runs in (("training", training), ("validation", validation)):
        times = [run["provenance"]["collection_seconds"] for run in runs]
        report[name + "_collection_seconds"] = sum(times) if all(value is not None for value in times) else None
    artifact = dict(
        version=VERSION,
        feature_version=VERSION,
        method="xgboost_config_features",
        label="log(latency_ms)",
        xgboost_version=xgb.__version__,
        params=params,
        best_rounds=best_rounds,
        schema=schema,
        domains=list(domains.values()),
        training=report,
        booster=json.loads(booster.save_raw(raw_format="json")),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


class Predictor:
    def __init__(self, path, *, expected_sha256=None, workers=4):
        _, xgb = _libraries()
        data = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("XGBoost model changed after the experiment request was frozen")
        self.artifact = artifact = json.loads(data)
        if (
            artifact.get("version") != VERSION
            or artifact.get("feature_version") != VERSION
            or artifact.get("method") != "xgboost_config_features"
        ):
            raise ValueError("unsupported XGBoost artifact/schema")
        self.workers = workers
        self.booster = xgb.Booster(params={"nthread": workers, "device": "cpu"})
        self.booster.load_model(bytearray(json.dumps(artifact["booster"]).encode()))
        self.booster.set_param({"nthread": workers, "device": "cpu"})

    def predict(self, context, configs):
        np, xgb = _libraries()
        artifact = self.artifact
        if domain(context) not in artifact["domains"]:
            raise ValueError("XGBoost model has no matching device, kernel implementation/source, and benchmark backend")
        seen = set(artifact["training"]["training_workloads"] + artifact["training"]["validation_workloads"])
        if workload_key(context) in seen:
            raise ValueError("evaluation workload was used for training or validation")
        data = xgb.DMatrix(_matrix([features(context, config) for config in configs], artifact["schema"]), nthread=self.workers)
        result = self.booster.predict(data, iteration_range=(0, artifact["best_rounds"]))
        if len(result) != len(configs) or not np.isfinite(result).all():
            raise ValueError("XGBoost returned invalid predictions")
        return [float(value) for value in result]

    def rank(self, context, configs, top_k):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("XGBoost requires a positive top-K measurement budget")
        if not configs or len({digest(config) for config in configs}) != len(configs):
            raise ValueError("XGBoost requires a nonempty grid of unique configurations")
        scores = self.predict(context, configs)
        order = sorted(range(len(configs)), key=lambda index: (scores[index], index))
        selected = order[:top_k]
        positions = defaultdict(list)
        for rank, index in enumerate(order, 1):
            positions[scores[index]].append(rank)
        ranking = [
            dict(
                index=index,
                score=scores[index],
                tier="eligible",
                rank=rank,
                tie_first_rank=min(positions[scores[index]]),
                tie_last_rank=max(positions[scores[index]]),
            )
            for rank, index in enumerate(order, 1)
        ]
        return dict(
            method="xgboost",
            model_sha256=self.sha256,
            model_training=self.artifact["training"],
            metric="xgboost_log_latency",
            score_units="log(ms)",
            ranking=ranking,
            selection=dict(
                requested_k=top_k,
                selected_indices=selected,
                selected_count=len(selected),
                shortfall=max(0, top_k - len(selected)),
                tie_break="original configuration index",
                failure_policy="no replacement after compilation or benchmark failure",
            ),
            configs=[
                dict(
                    index=i,
                    config=config,
                    selected=i in selected,
                    status="selected" if i in selected else "not_selected",
                    tile_cost=dict(score=scores[i], ranking_metric="xgboost_log_latency"),
                )
                for i, config in enumerate(configs)
            ],
        )


def evaluate(predictor, runs, top_k):
    results = []
    for run in runs:
        report = predictor.rank(run["context"], run["configs"], top_k)
        times = {sample["index"]: sample["latency_ms"] for sample in run["samples"]}
        selected = report["selection"]["selected_indices"]
        valid = [index for index in selected if index in times]
        results.append(
            dict(
                workload=run["context"]["workload"],
                provenance=run["provenance"],
                selection=report["selection"],
                selected_candidates_with_oracle_measurement=len(valid),
                oracle_at_k=min(times.values()) / min(times[index] for index in valid) if valid else None,
            )
        )
    return dict(model_sha256=predictor.sha256, top_k=top_k, results=results)

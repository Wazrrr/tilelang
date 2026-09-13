"""Small CLI adapter shared by the existing fixed-grid comparison runners."""

import hashlib
import json
from pathlib import Path
import time


def add_arguments(parser):
    parser.add_argument("--xgb-model", type=Path, help="Frozen XGBoost model; also adds XGBoost to --method all")
    parser.add_argument("--xgb-model-sha256", help="Expected model fingerprint, propagated to comparison workers")


def validate_arguments(parser, args):
    if args.method == "xgboost" and args.xgb_model is None:
        parser.error("--method xgboost requires --xgb-model")
    if args.xgb_model is not None:
        if args.top_k is None:
            parser.error("XGBoost requires a finite --top-k")
        args.xgb_model = args.xgb_model.resolve()
        actual = hashlib.sha256(args.xgb_model.read_bytes()).hexdigest()
        if args.xgb_model_sha256 is not None and actual != args.xgb_model_sha256:
            parser.error("XGBoost model fingerprint mismatch")
        args.xgb_model_sha256 = actual


def child_arguments(args):
    return ["--xgb-model", str(args.xgb_model), "--xgb-model-sha256", args.xgb_model_sha256] if args.xgb_model is not None else []


def rank_for_run(args, configs):
    from .data import context_from_experiment
    from .model import Predictor
    from experiments._common import write_json

    started = time.perf_counter()
    context = context_from_experiment(json.loads((args.output / "experiment.json").read_text()))
    predictor = Predictor(args.xgb_model, expected_sha256=args.xgb_model_sha256, workers=args.workers)
    report = predictor.rank(context, configs, args.top_k)
    seconds = time.perf_counter() - started
    report["selection"]["wall_time_ms"] = seconds * 1000
    write_json(args.output / "xgboost.json", report)
    return report, seconds

"""Train a frozen baseline or evaluate it using held-out exhaustive records."""

import argparse
import json
from pathlib import Path

from .data import read_runs
from .model import DEFAULT_LEARNING_RATE, DEFAULT_MAX_DEPTH, DEFAULT_ROUNDS, DEFAULT_SUBSAMPLE, Predictor, evaluate, train
from .sampling import DEFAULT_SAMPLE_FRACTION, SAMPLING_POLICY, SAMPLING_POLICIES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--train-runs", nargs="+", required=True, type=Path)
    training.add_argument("--validation-runs", nargs="+", required=True, type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of each training/validation configuration pool (default: 0.1)",
    )
    training.add_argument("--sampling-policy", choices=SAMPLING_POLICIES, default=SAMPLING_POLICY)
    training.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    training.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    training.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    training.add_argument(
        "--subsample", type=float, default=DEFAULT_SUBSAMPLE, help="Training-row fraction per boosting round (default: 0.8)"
    )
    training.add_argument("--seed", type=int, default=123)
    training.add_argument("--workers", type=int, default=4)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--runs", nargs="+", required=True, type=Path)
    evaluation.add_argument("--top-k", type=int, default=20)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        report = train(
            read_runs(args.train_runs),
            read_runs(args.validation_runs),
            args.output,
            sample_fraction=args.sample_fraction,
            sampling_policy=args.sampling_policy,
            rounds=args.rounds,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            seed=args.seed,
            workers=args.workers,
        )
    else:
        report = evaluate(Predictor(args.model), read_runs(args.runs), args.top_k)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

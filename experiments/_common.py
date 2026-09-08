"""Shared command-line inputs and output files for the final experiments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def run_name(value):
    if not value or value in (".", "..") or Path(value).name != value:
        raise argparse.ArgumentTypeError("must be a single directory name")
    return value


def add_run_arguments(parser, output):
    parser.add_argument("--output", type=Path, default=Path(output), help="Parent directory for saved runs")
    parser.add_argument(
        "--run-name", type=run_name, help="Run subdirectory name; default: current UTC timestamp. Existing names are rejected."
    )
    parser.add_argument("--workers", type=positive_int, default=4, help="CPU compilation workers")
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--rep", type=positive_int, default=100)
    parser.add_argument("--timeout", type=positive_int, default=30, help="Autotuner timeout per candidate, in seconds")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--config-indices", type=int, nargs="+", help="Explicit subset of the original grid; default: entire grid")


def add_gemm_arguments(parser):
    for name in ("m", "n", "k"):
        parser.add_argument(f"--{name}", type=positive_int, default=4096)


def prepare_run(args):
    """Create a separate result directory and disable caches before importing TileLang."""
    args.run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output = args.output / args.run_name
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(
            f"Run directory already exists: {args.output}. Choose another --run-name or omit it for a timestamped run."
        ) from None
    print(f"Results directory: {args.output}", flush=True)
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(args.workers)
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(args.output / "timings.tsv")


def select_configs(grid, indices):
    indices = list(range(len(grid))) if indices is None else indices
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(grid) for i in indices):
        raise ValueError(f"Config indices must be distinct integers in [0, {len(grid) - 1}]")
    return indices, [grid[i] for i in indices]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def device_info(devices):
    import torch

    return [
        {
            "ordinal": device,
            "name": torch.cuda.get_device_name(device),
            "uuid": str(torch.cuda.get_device_properties(device).uuid),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
        for device in devices
    ]


def source_hashes(kernel_source):
    root = Path(__file__).resolve().parents[1]
    paths = [*(root / "experiments").rglob("*.py"), *(root / "tilelang/tiletune").rglob("*.py"), root / kernel_source]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}

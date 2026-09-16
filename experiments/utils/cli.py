"""Shared configuration selection, device metadata and source fingerprints."""

import hashlib
from pathlib import Path


def select_configs(grid, indices):
    indices = list(range(len(grid))) if indices is None else indices
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(grid) for i in indices):
        raise ValueError(f"Config indices must be distinct integers in [0, {len(grid) - 1}]")
    return indices, [grid[i] for i in indices]


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


SOURCE_ROOTS = (
    "experiments/utils",
    "experiments/common",
    "experiments/gemm",
    "experiments/flash_attention",
    "experiments/kda",
    "experiments/softmax",
    "experiments/xgboost",
    "experiments/ascend",
    "examples/gemm",
    "examples/flash_attention",
    "examples/kda",
    "examples/online_softmax",
    "tiletune_core",
    "tilelang/tiletune",
    "tilelang/carver",
    "tilelang/autotuner",
)


def source_hashes(kernel_source, *extra_sources):
    """Fingerprint active code; generated results and archived sources are data."""
    root = Path(__file__).resolve().parents[2]
    paths = {root / name for name in (kernel_source, *extra_sources)}
    paths.update((root / "experiments").glob("*.py"))
    for directory in SOURCE_ROOTS:
        paths.update((root / directory).rglob("*.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def observe_compilation(execution, outcomes):
    """Retain each candidate's compile result before benchmark rows are available."""
    for future, items in execution[2].items():

        def record(done, items=items):
            try:
                results = done.result()
            except Exception as error:
                for index, _ in items:
                    outcomes[index] = dict(status="compilation_failed", error=str(error))
            else:
                for index, _, _, error in results:
                    outcomes[index] = dict(status="compilation_failed" if error else "compiled", error=str(error) if error else None)

        future.add_done_callback(record)
    return execution

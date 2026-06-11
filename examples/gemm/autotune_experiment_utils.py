from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


RESULT_COLUMNS = [
    "timestamp_utc",
    "example",
    "m",
    "n",
    "k",
    "use_autotune",
    "with_roller",
    "profile_backend",
    "use_pipeline",
    "enable_grouped_compile",
    "group_compile_size",
    "benchmark_multi_gpu",
    "benchmark_devices",
    "tilelang_config",
    "tilelang_latency",
    "ref_latency",
    "tilelang_tflops",
    "ref_tflops",
]


def make_seeded_gemm_inputs(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.empty((M, K), device=device, dtype=dtype).uniform_(-1.0, 1.0, generator=generator)
    b = torch.empty((N, K), device=device, dtype=dtype).uniform_(-1.0, 1.0, generator=generator)
    return [a, b]


def gemm_tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    return 2 * M * N * K / latency_ms * 1e-9


def append_autotune_result(
    results_tsv: str | None,
    *,
    example: str,
    M: int,
    N: int,
    K: int,
    use_autotune: bool,
    with_roller: bool,
    profile_backend: str,
    tilelang_config: dict[str, Any],
    tilelang_latency: float,
    ref_latency: float,
    use_pipeline: bool | str = "",
    enable_grouped_compile: bool | str = "",
    group_compile_size: int | str = "",
    benchmark_multi_gpu: bool | str = "",
    benchmark_devices: list[int] | None = None,
) -> None:
    if not results_tsv:
        return

    tilelang_tflops = gemm_tflops(M, N, K, tilelang_latency)
    ref_tflops = gemm_tflops(M, N, K, ref_latency)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "example": Path(example).name,
        "m": M,
        "n": N,
        "k": K,
        "use_autotune": use_autotune,
        "with_roller": with_roller,
        "profile_backend": profile_backend,
        "use_pipeline": use_pipeline,
        "enable_grouped_compile": enable_grouped_compile,
        "group_compile_size": group_compile_size,
        "benchmark_multi_gpu": benchmark_multi_gpu,
        "benchmark_devices": " ".join(str(device) for device in (benchmark_devices or [])),
        "tilelang_config": json.dumps(tilelang_config, sort_keys=True),
        "tilelang_latency": tilelang_latency,
        "ref_latency": ref_latency,
        "tilelang_tflops": tilelang_tflops,
        "ref_tflops": ref_tflops,
    }
    _append_tsv_row(Path(results_tsv), row)


def _append_tsv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    if not write_header:
        with path.open("r", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
        if header != RESULT_COLUMNS:
            raise ValueError(
                f"{path} has an incompatible header. "
                f"Expected {RESULT_COLUMNS}, got {header}."
            )

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})

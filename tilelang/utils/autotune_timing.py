"""Opt-in wall-clock timing for autotune orchestration.

Set ``TILELANG_AUTOTUNE_TIMING_LOG=/path/to/timing.tsv`` to append timing rows.
The helper is intentionally independent from TileLang imports so it can be used
from low-level CUDA compile callbacks without creating import cycles.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from collections.abc import Iterator


_TIMING_LOG_ENV = "TILELANG_AUTOTUNE_TIMING_LOG"
_LOCK = threading.Lock()
_HEADER = [
    "pid",
    "thread_id",
    "time_ns",
    "stage",
    "duration_ms",
    "group_size",
    "config_idx",
    "configs",
    "details",
]


def autotune_timing_log_path() -> str | None:
    path = os.environ.get(_TIMING_LOG_ENV)
    if path is None or path == "":
        return None
    return path


def is_autotune_timing_enabled() -> bool:
    return autotune_timing_log_path() is not None


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", "\\n")


def _json_cell(value: dict[str, Any] | None) -> str:
    if not value:
        return ""
    return _cell(json.dumps(value, sort_keys=True, default=str))


def record_autotune_timing(
    stage: str,
    duration_s: float,
    *,
    group_size: int | None = None,
    config_idx: int | None = None,
    configs: str | None = None,
    **details: Any,
) -> None:
    path_text = autotune_timing_log_path()
    if path_text is None:
        return

    path = Path(path_text)
    row = {
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "time_ns": time.time_ns(),
        "stage": stage,
        "duration_ms": duration_s * 1000.0,
        "group_size": group_size,
        "config_idx": config_idx,
        "configs": configs,
        "details": details,
    }

    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a") as file:
            if write_header:
                file.write("\t".join(_HEADER) + "\n")
            file.write(
                "\t".join(
                    [
                        _cell(row["pid"]),
                        _cell(row["thread_id"]),
                        _cell(row["time_ns"]),
                        _cell(row["stage"]),
                        _cell(f"{row['duration_ms']:.6f}"),
                        _cell(row["group_size"]),
                        _cell(row["config_idx"]),
                        _cell(row["configs"]),
                        _json_cell(row["details"]),
                    ]
                )
                + "\n"
            )


@contextmanager
def timed_autotune_stage(
    stage: str,
    *,
    group_size: int | None = None,
    config_idx: int | None = None,
    configs: str | None = None,
    **details: Any,
) -> Iterator[None]:
    if not is_autotune_timing_enabled():
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        record_autotune_timing(
            stage,
            time.perf_counter() - start,
            group_size=group_size,
            config_idx=config_idx,
            configs=configs,
            **details,
        )

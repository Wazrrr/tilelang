"""Parse CUDA PTXAS resource-usage output and expose it through a recorder."""

from __future__ import annotations

import contextlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KernelResourceUsage:
    """CUDA per-kernel resource counts reported by PTXAS."""

    n_regs: int = 0
    n_spills: int = 0
    scratch_bytes: int = 0
    n_max_threads: int | None = None
    static_smem_bytes: int = 0
    const_size_bytes: int = 0
    local_size_bytes: int = 0
    max_dynamic_smem_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_COMPILE_ENTRY_RE = re.compile(r"ptxas info\s*:\s*Compiling entry function '([^']+)'")
_FUNCTION_PROPERTIES_RE = re.compile(r"ptxas info\s*:\s*Function properties for\s+(.+?)\s*$")
_USED_RE = re.compile(r"ptxas info\s*:\s*Used\s+(?P<items>.+?)\s*$")
_BYTES_RE = re.compile(r"(?P<value>\d+)\s+bytes\s+(?P<kind>smem|cmem\[\d+\]|stack frame|spill stores|spill loads)")
_REGS_RE = re.compile(r"(?P<value>\d+)\s+registers")
_RECORDER = threading.local()
CUDA_RESOURCE_CAPTURE_CONFIG_KEY = "tl.enable_cuda_resource_capture"


def cuda_ptxas_verbose_flag() -> str:
    return "--ptxas-options=--verbose"


def reset_recorder() -> None:
    _RECORDER.items = {}


def is_capture_enabled() -> bool:
    return bool(getattr(_RECORDER, "capture_enabled", False))


@contextlib.contextmanager
def capture_resource_usage():
    previous = is_capture_enabled()
    _RECORDER.capture_enabled = True
    try:
        yield
    finally:
        _RECORDER.capture_enabled = previous


def pop_recorded() -> dict[str, KernelResourceUsage]:
    items = getattr(_RECORDER, "items", {})
    _RECORDER.items = {}
    return dict(items)


def record_usage(usage: dict[str, KernelResourceUsage]) -> None:
    items = getattr(_RECORDER, "items", None)
    if items is None:
        return
    items.update(usage)


def parse_ptxas_output(output: str) -> dict[str, KernelResourceUsage]:
    """Return exact PTXAS-reported resource usage keyed by kernel name."""
    usage: dict[str, KernelResourceUsage] = {}
    current_name: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        entry_match = _COMPILE_ENTRY_RE.search(line)
        if entry_match is not None:
            current_name = entry_match.group(1)
            usage.setdefault(current_name, KernelResourceUsage())
            continue

        props_match = _FUNCTION_PROPERTIES_RE.search(line)
        if props_match is not None:
            current_name = props_match.group(1).strip()
            usage.setdefault(current_name, KernelResourceUsage())
            continue

        if current_name is None:
            continue

        used_match = _USED_RE.search(line)
        if used_match is not None:
            item_text = used_match.group("items")
            item = usage.setdefault(current_name, KernelResourceUsage())
            regs_match = _REGS_RE.search(item_text)
            if regs_match is not None:
                item.n_regs = int(regs_match.group("value"))
                item.extra.setdefault("observed_fields", []).append("n_regs")
            for bytes_match in _BYTES_RE.finditer(item_text):
                value = int(bytes_match.group("value"))
                kind = bytes_match.group("kind")
                if kind == "smem":
                    item.static_smem_bytes = value
                elif kind.startswith("cmem"):
                    item.const_size_bytes += value
                    item.extra[kind] = value
            continue

        for bytes_match in _BYTES_RE.finditer(line):
            value = int(bytes_match.group("value"))
            kind = bytes_match.group("kind")
            item = usage.setdefault(current_name, KernelResourceUsage())
            if kind == "stack frame":
                item.local_size_bytes = max(item.local_size_bytes, value)
                item.extra.setdefault("observed_fields", []).append("local_size_bytes")
            elif kind == "spill stores":
                item.extra["spill_stores_bytes"] = value
                item.n_spills += value // 4
            elif kind == "spill loads":
                item.extra["spill_loads_bytes"] = value

    return usage


def filter_and_record(output: str) -> str:
    usage = parse_ptxas_output(output)
    if usage:
        record_usage(usage)
    return output


def dump_to_file(usage: dict[str, KernelResourceUsage], path: str) -> None:
    data = {name: asdict(u) for name, u in usage.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_from_file(path: str) -> dict[str, KernelResourceUsage]:
    with open(path) as f:
        data = json.load(f)
    out: dict[str, KernelResourceUsage] = {}
    for name, entry in data.items():
        with contextlib.suppress(TypeError):
            out[name] = KernelResourceUsage(
                n_regs=int(entry.get("n_regs", 0) or 0),
                n_spills=int(entry.get("n_spills", 0) or 0),
                scratch_bytes=int(entry.get("scratch_bytes", 0) or 0),
                n_max_threads=int(entry["n_max_threads"]) if entry.get("n_max_threads") is not None else None,
                static_smem_bytes=int(entry.get("static_smem_bytes", 0) or 0),
                const_size_bytes=int(entry.get("const_size_bytes", 0) or 0),
                local_size_bytes=int(entry.get("local_size_bytes", 0) or 0),
                max_dynamic_smem_bytes=int(entry["max_dynamic_smem_bytes"]) if entry.get("max_dynamic_smem_bytes") is not None else None,
                extra=dict(entry.get("extra", {})),
            )
    return out

"""Backend-neutral kernel resource-usage serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class GenericKernelResourceUsage:
    """Resource counts shared by CUDA and HIP resource recorders."""

    n_regs: int = 0
    n_spills: int = 0
    scratch_bytes: int = 0
    n_max_threads: int | None = None
    static_smem_bytes: int = 0
    const_size_bytes: int = 0
    local_size_bytes: int = 0
    max_dynamic_smem_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if is_dataclass(usage):
        data = asdict(usage)
    elif isinstance(usage, dict):
        data = dict(usage)
    else:
        data = {
            key: value
            for key, value in vars(usage).items()
            if not key.startswith("_")
        }

    for key in (
        "n_regs",
        "n_spills",
        "scratch_bytes",
        "static_smem_bytes",
        "const_size_bytes",
        "local_size_bytes",
    ):
        data[key] = int(data.get(key, 0) or 0)

    if data.get("n_max_threads") is not None:
        data["n_max_threads"] = int(data["n_max_threads"])
    if data.get("max_dynamic_smem_bytes") is not None:
        data["max_dynamic_smem_bytes"] = int(data["max_dynamic_smem_bytes"])

    data["extra"] = dict(data.get("extra", {}))
    data["backend"] = data.get("backend", usage.__class__.__module__.split(".")[-1] if not isinstance(usage, dict) else "generic")
    return data


def usage_to_json_dict(usage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: _usage_to_dict(item) for name, item in usage.items()}


def usage_from_json_dict(data: dict[str, Any]) -> dict[str, GenericKernelResourceUsage]:
    out: dict[str, GenericKernelResourceUsage] = {}
    for name, entry in data.items():
        out[name] = GenericKernelResourceUsage(
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


def dump_to_file(usage: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(usage_to_json_dict(usage), f, indent=2, sort_keys=True)


def load_from_file(path: str) -> dict[str, GenericKernelResourceUsage]:
    with open(path) as f:
        return usage_from_json_dict(json.load(f))

"""Read oracle records and evaluate saved selections without accelerator imports."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

TERMINAL = {
    "benchmarked",
    "compilation_failed",
    "elaboration_failed",
    "analysis_failed",
    "pre_lowering_rejected",
    "post_compile_rejected",
    "benchmark_error",
    "benchmark_timeout",
    "worker_failed",
}
UNAVAILABLE = {"unsupported", "unavailable", "model_unavailable", "failed"}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def config_key(config):
    if not isinstance(config, dict) or not config:
        raise ValueError("each candidate needs a nonempty config dictionary")
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(path):
    return json.loads(path.read_text())


def provenance(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def records_by_index(records):
    if not isinstance(records, list) or not records:
        raise ValueError("expected a nonempty candidate-record list")
    indexed, keys = {}, set()
    for row in records:
        index = row.get("index")
        if type(index) is not int or index < 0 or index in indexed:
            raise ValueError(f"invalid or duplicate candidate index: {index!r}")
        key = config_key(row.get("config"))
        if key in keys:
            raise ValueError("duplicate configuration in candidate records")
        indexed[index] = row
        keys.add(key)
    return indexed


def load_oracle(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = next((path / n for n in ("oracle.json", "outcomes.json", "brute_force.json") if (path / n).is_file()), path)
    data = read(path)
    sources = [provenance(path)]
    # Heuristic winner files point to the full sweep. Do not mix their later
    # winner-validation latency with the sweep's per-candidate measurements.
    if isinstance(data, dict) and "reference" in data:
        ref = data["reference"]
        target = Path(ref["path"])
        path = target if target.is_absolute() else path.parent / target
        source = provenance(path)
        if source["sha256"] != ref["sha256"]:
            raise ValueError("oracle reference SHA256 mismatch")
        sources.append(source)
        data = read(path)
    records = data if isinstance(data, list) else data.get("records", data.get("configs"))
    indexed = records_by_index(records)
    if isinstance(data, dict) and data.get("candidate_count", len(records)) != len(records):
        raise ValueError("incomplete oracle: candidate_count does not match records")
    # When available, the run manifest also catches missing failed candidates.
    manifest = path.parent / "experiment.json"
    if manifest.is_file():
        declared = read(manifest)["configs"]
        if len(declared) != len(records) or {config_key(c) for c in declared} != {config_key(r["config"]) for r in records}:
            raise ValueError("incomplete oracle: records do not match experiment.json")
    for row in indexed.values():
        if row.get("status") not in TERMINAL:
            raise ValueError(f"incomplete oracle outcome at index {row['index']}: {row.get('status')!r}")
        if row["status"] == "benchmarked" and (not finite(row.get("latency_ms")) or row["latency_ms"] <= 0):
            raise ValueError(f"invalid oracle latency at index {row['index']}")
    valid = [r for r in records if r["status"] == "benchmarked"]
    if not valid:
        raise ValueError("oracle has no correct measured candidates")
    winner = min(valid, key=lambda r: (r["latency_ms"], r["index"]))
    return dict(
        sources=sources,
        path=path,
        records={config_key(r["config"]): r for r in records},
        winner=winner,
        statuses=dict(Counter(r["status"] for r in records)),
    )


def metadata(path):
    data = read(path)
    if isinstance(data, dict) and "identity" in data:
        return data["identity"]
    manifest = path.parent / "experiment.json"
    if not manifest.is_file():
        return {}
    data = read(manifest)
    return dict(
        workload=data["workload"],
        target=data.get("device", {}).get("target", data.get("target")),
        device=data.get("device_observation", {}).get("name"),
        native_build=data.get("native_build"),
        measurement_identity=data.get("measurement_identity"),
        kernel_sources=data.get("source_sha256"),
    )


def check_metadata(oracle_path, method_path):
    """Compare recorded identity fields without importing backend definitions."""
    a, b = metadata(oracle_path), metadata(method_path)
    if not a or not b:
        return "not available; caller must supply the same workload and measurement domain"
    checked = []
    if isinstance(a.get("workload"), dict) and isinstance(b.get("workload"), dict):
        aw, bw = a["workload"], b["workload"]
        # Identity records can spell out defaults absent in older manifests.
        # Check every shared parameter without inventing operation-specific defaults.
        for left, right in ((aw, bw), (aw.get("parameters", {}), bw.get("parameters", {}))):
            keys = left.keys() & right.keys() - {"name", "configs", "config_space", "parameters"}
            if any(left[k] != right[k] for k in keys):
                raise ValueError(f"{method_path}: workload differs from the oracle run")
        checked.append("shared workload fields")
    compatible = a.get("measurement_identity") is not None and b.get("measurement_identity") is not None
    if compatible and a["measurement_identity"] != b["measurement_identity"]:
        raise ValueError(f"{method_path}: measurement identity differs from the oracle run")
    if compatible:
        checked.append("measurement identity (kernel/compiler/runtime/settings)")
    for key in ("target", "device") if compatible else ("target", "device", "native_build"):
        if a.get(key) is not None and b.get(key) is not None:
            if a[key] != b[key]:
                raise ValueError(f"{method_path}: {key} differs from the oracle run")
            checked.append(key)
    if not compatible and a.get("kernel_sources") and b.get("kernel_sources"):
        keys = a["kernel_sources"].keys() & b["kernel_sources"].keys()
        if any(a["kernel_sources"][k] != b["kernel_sources"][k] for k in keys):
            raise ValueError(f"{method_path}: kernel source hashes differ from the oracle run")
        if keys:
            checked.append("shared source hashes")
    return "matched " + ", ".join(checked) if checked else "no comparable metadata fields"


def validate_order(indices, records):
    if not isinstance(indices, list) or any(type(i) is not int or i not in records for i in indices):
        raise ValueError("selection/ranking references an unknown candidate index")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate in selection/ranking")
    return indices


def measure_prefix(indices, records, oracle, k, source):
    selected = indices[:k]
    measured = [(i, oracle["records"][config_key(records[i]["config"])]) for i in selected]
    valid = [(i, r) for i, r in measured if r["status"] == "benchmarked"]
    best = min(valid, key=lambda pair: (pair[1]["latency_ms"], pair[1]["index"])) if valid else None
    best_ms = best[1]["latency_ms"] if best else None
    # Same workload: throughput is inversely proportional to kernel latency.
    ratio = oracle_at_k(
        {r["index"]: r["latency_ms"] for r in oracle["records"].values() if r["status"] == "benchmarked"},
        [r["index"] for _, r in measured],
    )
    return dict(
        source=source,
        k=k,
        selected_count=len(selected),
        shortfall=max(0, k - len(selected)),
        successful_count=len(valid),
        failed_count=len(selected) - len(valid),
        status="no_success" if not valid else "shortfall" if len(selected) < k else "ok",
        selected_indices=selected,
        oracle_indices=[r["index"] for _, r in measured],
        best_index=best[0] if best else None,
        best_oracle_index=best[1]["index"] if best else None,
        best_config=best[1]["config"] if best else None,
        best_latency_ms=best_ms,
        oracle_at_k=ratio,
        oracle_percent=100 * ratio if ratio is not None else None,
        latency_gap_percent=100 * (best_ms / oracle["winner"]["latency_ms"] - 1) if best else None,
    )


def oracle_at_k(times, indices):
    """Retained throughput from a validated latency table; failures consume K."""
    available = [times[i] for i in indices if i in times]
    return min(times.values()) / min(available) if times and available else None

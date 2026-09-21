"""Load the intersection of schedules actually compiled for the H200 cases."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def pool_digest(configs):
    return hashlib.sha256(json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compiled_configs(family, candidates):
    """Reject changed grids instead of admitting unqualified new schedules.

    The certificate records compilation for the five final workloads on SM90a.
    It makes no correctness, performance, or other-target guarantee.
    """
    path = ROOT / "experiments/compilation" / f"{family}.json"
    certificate = json.loads(path.read_text())
    if certificate["candidate_pool_sha256"] != pool_digest(candidates):
        raise ValueError(f"{family} candidate grid changed; repeat H200 compilation qualification")
    indices = certificate["accepted_indices"]
    if certificate["candidate_count"] != len(candidates) or certificate["compiled_count"] != len(indices):
        raise ValueError(f"{family} compilation certificate has inconsistent counts")
    if indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < len(candidates) for i in indices):
        raise ValueError(f"{family} compilation certificate contains invalid indices")
    for filename, expected in certificate["kernel_sources"].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{family} compilation source changed: {filename}; repeat qualification")
    return [candidates[i] for i in indices]


def publish_compiled_pool(family, output, candidates, originals):
    """Publish only a complete, verified intersection retaining the example grid."""
    from experiments.common.spaces import config_id
    from experiments.families import FAMILIES, family_module
    from experiments.utils.baseline_store import EXAMPLES
    from experiments.utils.io import write_json

    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((output / "summary.json").read_text())
    op = next(op for op, name in FAMILIES.items() if name == family)
    expected_workloads = [w.to_dict() for w in family_module(op, "cases").cases(holdout=True)]
    if manifest["target"] != {"kind": "cuda", "arch": "sm_90a"}:
        raise ValueError("H200 qualification requires the SM90a target")
    if manifest["workloads"] != expected_workloads or summary["workloads"] != expected_workloads:
        raise ValueError("compilation evidence must cover all current final workloads")
    if summary["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("compilation manifest differs from the completed result")
    if candidates != manifest["configs"]:
        raise ValueError("candidate pool differs from compilation inputs")
    accepted = summary["configs"]
    if len(accepted) <= 500 or any(c not in accepted for c in originals):
        raise ValueError("compiled pool must exceed 500 and contain every original example config")
    if any(c not in candidates for c in accepted) or len({config_id(c) for c in accepted}) != len(accepted):
        raise ValueError("compiled result contains duplicate or undeclared configurations")
    outcomes = {}
    for workload in manifest["workloads"]:
        records = []
        for c in accepted:
            path = output / workload["name"] / f"{config_id(c)}.json"
            row = json.loads(path.read_text())
            if row["config"] != c or row["status"] != "compiled" or len(row["cuda_sha256"]) != 64:
                raise ValueError(f"missing successful compilation evidence: {path}")
            records.append(dict(config_id=config_id(c), cuda_sha256=row["cuda_sha256"]))
        outcomes[workload["name"]] = dict(compiled_count=len(records), records_sha256=pool_digest(records))
    examples = {EXAMPLES[family]}
    if family == "gemm_fp8":
        examples.add("examples/gemm_fp8/example_gemm_fp8_tiletune.py")
    # Runtime input/numerical-check utilities are recorded in the full evidence
    # manifest; they are not generated CUDA kernel sources.
    source_names = examples | {f"experiments/{family}/kernel.py", f"experiments/{family}/cases.py"}
    sources = {name: manifest["sources"][name] for name in sorted(source_names)}
    for name, value in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value:
            raise ValueError(f"kernel source changed after compilation: {name}")
    certificate = dict(
        version=1,
        target=manifest["target"],
        scope="Device compilation for all five final workloads; no GPU execution or correctness/performance claim",
        candidate_count=len(candidates),
        compiled_count=len(accepted),
        original_count=len(originals),
        original_pool_included=True,
        candidate_pool_sha256=pool_digest(candidates),
        accepted_indices=[i for i, c in enumerate(candidates) if c in accepted],
        kernel_sources=sources,
        compiler_libraries={k: v for k, v in manifest["sources"].items() if k.startswith("build/lib/")},
        nvcc=manifest["nvcc"],
        workloads=manifest["workloads"],
        outcomes=outcomes,
        evidence_directory=str(output.resolve()),
        evidence_manifest_sha256=summary["manifest_sha256"],
    )
    directory = ROOT / "experiments/compilation"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"{family}.json", certificate)
    return certificate

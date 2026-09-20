"""Immutable baseline bundles, keyed independently of TileTune's ranking code."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.utils.io import write_json
from experiments.utils.results import config_key, load_oracle, records_by_index, validate_order
from experiments.common.spec import Workload, configuration_space
from experiments.families import FAMILIES
from experiments.xgboost.data import digest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "gemm": "examples/gemm/example_gemm_advanced_autotune.py",
    "grouped_gemm": "examples/grouped_gemm/example_grouped_gemm_fwd.py",
    "flash_attention": "examples/flash_attention/example_mha_fwd_bshd.py",
    "kda": "examples/kda/chunk_intra_token_parallel.py",
    "gemm_fp8": "examples/gemm_fp8/example_blockscaled_gemm.py",
}


def hash_files(paths, root=ROOT):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}


def measurement_sources(families, root=ROOT):
    # TileTune and tiletune_core only rank candidates; changes there do not
    # change the baseline kernels. Keep compiler/JIT/profiler provenance even
    # though it is not a condition for reusing the fixed baseline measurements.
    paths = []
    for directory in ("src", "tilelang", "cmake"):
        paths += [
            p
            for p in (root / directory).rglob("*")
            if p.suffix in (".py", ".cc", ".h", ".cuh", ".cu", ".cmake")
            and "tiletune" not in p.relative_to(root).parts
            and "carver" not in p.relative_to(root).parts
        ]
    paths += [
        root / p
        for p in (
            "CMakeLists.txt",
            "experiments/backend.py",
            "experiments/utils/kernel.py",
            "experiments/utils/cli.py",
            "experiments/families.py",
            "experiments/common/kernels.py",
            "experiments/common/spec.py",
            "experiments/common/spaces.py",
            "experiments/common/run.py",
            "experiments/common/execution.py",
            "experiments/utils/monitor.py",
        )
    ]
    for family in families:
        paths += [root / f"experiments/{family}/{name}.py" for name in ("kernel", "reference", "spaces")]
        paths.append(root / EXAMPLES[family])
    return hash_files(paths, root)


def runtime_identity(device):
    """Observe the local device and toolchain without launching a workload."""
    if device.worker or device.target["kind"] != "cuda":
        raise ValueError("reusable baseline collection currently requires a local CUDA worker")
    # A short-lived probe avoids leaving a coordinator CUDA context that later
    # measurement workers would correctly identify as a foreign process.
    code = """import json, sys
from contextlib import redirect_stdout
from experiments.common.spec import Device
from experiments.utils.baseline_store import _runtime_identity
with redirect_stdout(sys.stderr):
    identity = _runtime_identity(Device(**json.loads(sys.argv[1])))
print(json.dumps(identity))
"""
    return json.loads(subprocess.check_output([sys.executable, "-c", code, json.dumps(device.to_dict())], text=True))


def _runtime_identity(device):
    import torch
    from tilelang.contrib.cc import get_cplus_compiler
    from tilelang.contrib.nvcc import find_cuda_path
    from tilelang.tiletune import current_target
    from experiments.common.run import targets_match
    from experiments.utils.monitor import snapshot, visible_gpus

    prop = torch.cuda.get_device_properties(0)
    if not targets_match(device.target, current_target()):
        raise ValueError("requested target differs from the visible accelerator")
    gpu = visible_gpus(snapshot())[0]
    compiler = get_cplus_compiler()
    nvcc = str(Path(find_cuda_path()) / "bin/nvcc")
    dependencies = {}
    for path in (ROOT / "3rdparty").iterdir():
        if not (path / ".git").exists():
            continue
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(path), "diff", "--binary", "HEAD"])
        extra = subprocess.check_output(["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
        dependencies[path.name] = dict(
            commit=head, diff_sha256=hashlib.sha256(diff).hexdigest(), untracked=hash_files([path / p for p in extra])
        )
    cache = ROOT / "build/CMakeCache.txt"
    build_options = (
        [
            line
            for line in cache.read_text().splitlines()
            if line and not line.startswith(("#", "//")) and (line.startswith(("CMAKE_C", "CMAKE_BUILD_TYPE", "USE_", "TL_")))
        ]
        if cache.exists()
        else []
    )
    return dict(
        device=gpu["name"],
        sm_count=prop.multi_processor_count,
        memory_bytes=prop.total_memory,
        target=device.target,
        python=sys.version,
        torch=torch.__version__,
        runtime=torch.version.cuda,
        driver=subprocess.check_output(
            ["nvidia-smi", "-i", gpu["uuid"], "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        host_compiler=subprocess.check_output([compiler, "--version"], text=True).strip(),
        device_compiler=subprocess.check_output([nvcc, "--version"], text=True).strip(),
        dependencies=dependencies,
        build_options=build_options,
    )


def identities(plan, device, settings, runtime, baseline_seed=123):
    workloads = [Workload(**w) for values in plan["splits"].values() for w in values]
    families = sorted({FAMILIES[w.op] for w in workloads})
    measurement = dict(
        version=1,
        # Only KDA changed semantics; keep compatible baseline bundles for
        # the four unchanged families reusable under contract version 2.
        kernel_contract_version=3 if "kda" in families else 2,
        sources=measurement_sources(families),
        runtime=runtime,
        timing={k: settings[k] for k in ("warmup", "rep", "timeout", "workers", "case_timeout")},
        backend="event",
        memory_regime="streaming",
        cache_flush_bytes=268435456,
        input_seed=123,
    )
    pools = {}
    for w in workloads:
        space = configuration_space(w, device)
        indices = (device.subsets or {}).get(w.name, list(range(len(space["configs"]))))
        pools[w.name] = [space["configs"][i] for i in indices]
    baseline_paths = [
        *(ROOT / "experiments/xgboost").glob("*.py"),
        *(ROOT / "tilelang/carver").rglob("*.py"),
        *(ROOT / "experiments").glob("*/carver.py"),
        ROOT / "experiments/common/baselines.py",
        ROOT / "experiments/common/comparison.py",
        ROOT / "experiments/utils/baseline_store.py",
    ]
    import xgboost

    baseline = dict(
        version=1,
        measurement=measurement,
        splits=plan["splits"],
        pools=pools,
        baseline_seed=baseline_seed,
        xgboost=plan["xgboost"],
        xgboost_version=xgboost.__version__,
        baseline_top_k=20,
        sources=hash_files(baseline_paths),
    )
    return measurement, baseline


def verify_bundle(path, identity):
    manifest = json.loads((path / "complete.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("baseline bundle identity differs")
    for name, expected in manifest["artifacts"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"baseline artifact changed: {name}")
    return manifest


def reuse_identity(identity):
    """Identify the measured cases, independently of collection provenance."""
    runtime = identity.get("measurement", {}).get("runtime", {})
    cases = {}
    for item in identity["splits"]["test"]:
        workload = Workload(**item)
        cases[workload.name] = dict(
            op=workload.op,
            parameters=workload.parameters,
            dtype=workload.dtype,
            configs=sorted(config_key(config) for config in identity["pools"][workload.name]),
        )
    return dict(
        gpu={key: runtime.get(key) for key in ("device", "target")},
        cases=cases,
        kernel_contract_version=identity.get("measurement", {}).get("kernel_contract_version"),
    )


def storage_root(family, device, runtime, override=None):
    """Keep each family's baseline records beside its experiment definition."""
    if override is not None:
        return Path(override).resolve() / device.name / family
    gpu = runtime["device"].removeprefix("NVIDIA ").replace("/", "_")
    return ROOT / "experiments" / family / "results" / gpu / "baselines"


def load_bundle(root, identity, *, reference=None):
    """Read the pinned study reference, or current.json for a new TileTune run."""
    root = Path(root)
    if reference is None:
        current = root / "current.json"
        if not current.is_file():
            raise FileNotFoundError(f"No saved baseline bundle in {root}; run the family command with --run-baselines first")
        reference = json.loads(current.read_text())
        path = root / reference["path"]
    else:
        path = Path(reference["path"])
    if hashlib.sha256((path / "complete.json").read_bytes()).hexdigest() != reference["manifest_sha256"]:
        raise ValueError(f"baseline completion manifest changed: {path}")
    recorded = json.loads((path / "complete.json").read_text())["identity"]
    if "identity_sha256" in reference and reference["identity_sha256"] != digest(recorded):
        raise ValueError(f"baseline identity reference changed: {path}")
    saved, requested = reuse_identity(recorded), reuse_identity(identity)
    if (
        saved["kernel_contract_version"] != requested["kernel_contract_version"]
        or saved["gpu"] != requested["gpu"]
        or any(saved["cases"].get(name) != case for name, case in requested["cases"].items())
    ):
        raise ValueError(
            f"Saved baselines in {root} do not match the kernel contract, workload, GPU or configuration pool; explicitly rerun with --run-baselines"
        )
    verify_bundle(path, recorded)
    return path, True


def publish_bundle(root, path, identity):
    """Called only by explicit collection, after the complete bundle is verified."""
    write_json(
        root / "current.json",
        dict(
            path=str(path.relative_to(root)),
            identity_sha256=digest(identity),
            manifest_sha256=hashlib.sha256((path / "complete.json").read_bytes()).hexdigest(),
        ),
    )


def validate_case_bundle(case, configs):
    """Require the exact declared pool and a complete permutation of its ranking."""
    oracle = load_oracle(case / "brute_force")
    expected = {config_key(c) for c in configs}
    if set(oracle["records"]) != expected:
        raise ValueError("baseline oracle pool differs from the declared configurations")
    for method in ("carver", "xgboost"):
        result = json.loads((case / method / "result.json").read_text())
        if result["status"] not in ("completed", "unsupported", "model_unavailable"):
            raise ValueError(f"baseline {method} failed for {case.name}: {result}")
        if result["status"] != "completed" and method != "carver":
            raise ValueError("XGBoost baseline is required")
        if result["status"] == "unsupported":
            continue
        report = json.loads((case / method / (method + ".json")).read_text())
        records = records_by_index(report["configs"])
        if {config_key(r["config"]) for r in records.values()} != expected:
            raise ValueError(f"{method} pool differs from the declared configurations")
        ranked = validate_order([r["index"] for r in report["ranking"]], records)
        if len(ranked) != len(records):
            raise ValueError(f"incomplete {method} ranking")
        selected = validate_order(report["selection"]["selected_indices"], records)
        if result["status"] == "model_unavailable":
            if (
                selected
                or any(r["status"] != "model_rejected" for r in records.values())
                or any(r["score"] is not None or r["tier"] == "eligible" for r in report["ranking"])
            ):
                raise ValueError("unavailable Carver baseline must record rejection of the complete pool")
        elif (
            result["selection"] != report["selection"]
            or result["winner"]["index"] not in selected
            or config_key(result["winner"]["config"]) != config_key(records[result["winner"]["index"]]["config"])
        ):
            raise ValueError(f"{method} result does not match its frozen selection")


@contextmanager
def bundle_lock(root, key):
    root.mkdir(parents=True, exist_ok=True)
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield root / key
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def collect_bundle(root, identity, measurement, device, settings, *, refresh=False, run=subprocess.run):
    """Explicit collection; publish a new current reference only after success.

    A refresh keeps prior bundles intact so earlier TileTune runs still refer to
    their original measurements. A failed refresh leaves current.json unchanged.
    """
    root = Path(root).resolve()
    key = digest(identity)
    with bundle_lock(root, "collection"):
        import time

        path = root / "runs" / (f"{key}.{time.time_ns()}" if refresh else key)
        if (path / "complete.json").exists():
            verify_bundle(path, identity)
            publish_bundle(root, path, identity)
            return path, True
        if path.exists():
            path.rename(path.with_name(path.name + f".incomplete.{time.time_ns()}"))
        path.mkdir(parents=True)
        write_json(path / "identity.json", identity)
        write_json(path / "measurement.json", measurement)
        write_json(path / "splits.json", dict(version=1, devices=[device.to_dict()], splits=identity["splits"]))
        command = [
            sys.executable,
            "-m",
            "experiments.common.comparison",
            "--split-manifest",
            str(path / "splits.json"),
            "--methods",
            "carver",
            "xgboost",
            "--top-k",
            "20",
            "--budget-fraction",
            "1",
            "--xgb-sample-fraction",
            str(identity.get("xgboost", {}).get("sample_fraction", 0.1)),
            "--seed",
            str(identity["baseline_seed"]),
            "--measurement-identity",
            str(path / "measurement.json"),
            "--skip-profile",
            "--skip-validation",
            "--wait-idle",
            "--output",
            str(path / "collection"),
        ]
        for name, value in settings.items():
            command.extend(("--" + name.replace("_", "-"), str(value)))
        run(command, check=True)
        cases = path / "collection" / device.name / "test"
        for w in identity["splits"]["test"]:
            validate_case_bundle(cases / w["name"], identity["pools"][w["name"]])
        artifacts = hash_files(path.rglob("*.json"), path)
        write_json(path / "complete.json", dict(version=1, identity=identity, artifacts=artifacts))
        publish_bundle(root, path, identity)
        return path, False

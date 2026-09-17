"""Immutable baseline bundles, keyed independently of TileTune's ranking code."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.utils.io import write_json
from experiments.utils.results import load_oracle
from experiments.common.spec import Workload, configuration_space
from experiments.families import FAMILIES
from experiments.xgboost.data import digest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "gemm": "examples/gemm/example_gemm_advanced_autotune.py",
    "flash_attention": "examples/flash_attention/example_mha_fwd_bshd.py",
    "kda": "examples/kda/chunk_o.py",
    "gemm_fp8": "examples/gemm_fp8/example_tilelang_gemm_fp8.py",
}


def hash_files(paths, root=ROOT):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file()}


def measurement_sources(families, root=ROOT):
    # TileTune and tiletune_core only rank candidates; changes there do not
    # change the baseline kernels. Compiler/JIT/profiler changes do invalidate.
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
    # Pool subsets can exceed the OS limit for a single argv value. Runtime
    # observation needs only target identity, never thousands of config IDs.
    description = dict(name=device.name, target=device.target)
    return json.loads(subprocess.check_output([sys.executable, "-c", code, json.dumps(description)], text=True))


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
        ROOT / "experiments/gemm/carver.py",
        ROOT / "experiments/common/baselines.py",
        ROOT / "experiments/common/carver_graph.py",
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


@contextmanager
def bundle_lock(root, key):
    root.mkdir(parents=True, exist_ok=True)
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield root / key
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def collect_bundle(root, identity, measurement, device, settings, *, run=subprocess.run):
    """Collect once. Completed bundles are verified and never written on reuse."""
    key = digest(identity)
    with bundle_lock(Path(root), key) as path:
        if (path / "complete.json").exists():
            verify_bundle(path, identity)
            return path, True
        if path.exists():
            import time

            path.rename(path.with_name(path.name + f".incomplete.{time.time_ns()}"))
        path.mkdir()
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
            case = cases / w["name"]
            oracle = load_oracle(case / "brute_force")
            if len(oracle["records"]) != len(identity["pools"][w["name"]]):
                raise ValueError("baseline oracle pool is incomplete")
            for method in ("carver", "xgboost"):
                result = json.loads((case / method / "result.json").read_text())
                if result["status"] not in ("completed", "unsupported"):
                    raise ValueError(f"baseline {method} failed for {w['name']}: {result}")
                if result["status"] == "unsupported" and method != "carver":
                    raise ValueError("XGBoost baseline is required")
                if result["status"] == "completed" and len(json.loads((case / method / (method + ".json")).read_text())["ranking"]) != len(
                    oracle["records"]
                ):
                    raise ValueError(f"incomplete {method} ranking")
        artifacts = hash_files(path.rglob("*.json"), path)
        write_json(path / "complete.json", dict(version=1, identity=identity, artifacts=artifacts))
        return path, False

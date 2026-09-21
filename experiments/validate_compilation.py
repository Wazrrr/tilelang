"""Resume CPU-only SM90a compilation checks, one workload at a time.

This lowers the real experiment PrimFunc and invokes NVCC/PTXAS. It never
creates CUDA tensors, loads a kernel on a GPU, or measures correctness/speed.
Failed candidates are excluded from subsequent workloads; the final intersection
contains only candidates compiled successfully for every declared final shape.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

from experiments.families import FAMILIES, family_module
from experiments.common.spaces import config_id
from experiments.common.spec import Workload
from experiments.utils.io import write_json
from experiments.utils.isolation import measurement_lease, select_cpu_ids


TARGET = {"kind": "cuda", "arch": "sm_90a"}


def original_configs(family, workload):
    if family == "gemm":
        from examples.gemm.example_gemm_advanced_autotune import get_configs

        return get_configs(*(workload.parameters[k] for k in ("m", "n", "k")))
    if family == "gemm_fp8":
        from examples.gemm_fp8.example_gemm_fp8_tiletune import get_configs
    elif family == "flash_attention":
        from examples.flash_attention.example_mha_fwd_bshd import get_configs
    elif family == "kda":
        from examples.kda.chunk_intra_token_parallel import get_configs
    elif family == "grouped_gemm":
        return [dict(block_M=64, block_N=128, block_K=64, num_stages=2, threads=256)]
    else:
        raise ValueError(f"unknown family {family}")
    return get_configs()


def initialize_worker(cpu_ids, log_dir):
    os.sched_setaffinity(0, cpu_ids)
    # Keep compiler diagnostics out of coordinator output; individual failures
    # also retain their exception text in the resumable result record.
    stream = open(Path(log_dir) / f"worker-{os.getpid()}.log", "a")
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    from experiments.utils.imports import use_local_tilelang

    use_local_tilelang()


def compile_candidate(workload, config):
    from experiments.common.kernels import make_case
    from tilelang import tvm
    from tilelang.engine import lower

    started = time.monotonic()
    try:
        case = make_case(Workload(**workload))
        func = case.build(**config)
        target = tvm.target.Target(TARGET)
        with target, tvm.transform.PassContext(opt_level=3, config=case.pass_configs):
            artifact = lower(func, target=target, enable_device_compile=True)
        result = dict(status="compiled", cuda_sha256=hashlib.sha256(artifact.kernel_source.encode()).hexdigest())
    except Exception as exc:
        # This is the batch experiment boundary: a failure is data, never a
        # successful configuration or a substitute configuration.
        result = dict(status="failed", error_type=type(exc).__name__, error=str(exc))
    return dict(result, seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILIES.values()), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--publish", action="store_true", help="publish a complete pool only after checking original-grid inclusion")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("set CUDA_VISIBLE_DEVICES='' for CPU-only compilation validation")
    if args.workers < 1:
        parser.error("workers must be positive")
    root = Path(__file__).resolve().parents[1]
    from experiments.utils.imports import use_local_tilelang

    use_local_tilelang()
    from tilelang.contrib.nvcc import find_cuda_path

    op = next(op for op, family in FAMILIES.items() if family == args.family)
    workloads = family_module(op, "cases").cases(holdout=True)
    configs = family_module(op, "spaces").candidate_configs()
    paths = [root / f"experiments/{args.family}/{name}.py" for name in ("cases", "kernel", "spaces")]
    paths += [root / "experiments/utils/kernel.py", Path(__file__)]
    paths += sorted((root / "examples" / ("flash_attention" if args.family == "flash_attention" else args.family)).glob("*.py"))
    paths += sorted((root / "tilelang").rglob("*.py"))
    paths += sorted((root / "src").rglob("*.cc"))
    paths += sorted((root / "src").rglob("*.h"))
    paths += sorted((root / "src/tl_templates").rglob("*.cuh"))
    paths += sorted((root / "build/lib").glob("*.so"))
    identity = dict(
        target=TARGET,
        workloads=[asdict(w) for w in workloads],
        configs=configs,
        sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        nvcc=subprocess.check_output([str(Path(find_cuda_path()) / "bin/nvcc"), "--version"], text=True),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("compilation inputs changed; use a new output directory")
    else:
        write_json(manifest, identity)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    accepted = {config_id(c) for c in configs}
    with measurement_lease([]):
        cpu_ids = select_cpu_ids(args.workers)
        write_json(args.output / "execution.json", dict(workers=args.workers, cpu_ids=cpu_ids, gpu_execution=False))
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(cpu_ids, str(logs)),
        ) as executor:
            for w in workloads:
                output = args.output / w.name
                output.mkdir(exist_ok=True)
                pending = {}
                checked = []
                for c in configs:
                    key = config_id(c)
                    if key not in accepted:
                        continue
                    path = output / f"{key}.json"
                    if path.exists():
                        checked.append(json.loads(path.read_text()))
                    else:
                        pending[executor.submit(compile_candidate, asdict(w), c)] = (key, c, path)
                print(f"{w.name}: {len(pending)} pending, {len(checked)} cached", flush=True)
                for future in as_completed(pending):
                    key, c, path = pending[future]
                    row = dict(config_id=key, config=c, **future.result())
                    write_json(path, row)
                    checked.append(row)
                    if len(checked) % 128 == 0:
                        print(f"{w.name}: {len(checked)} checked, {sum(r['status'] == 'failed' for r in checked)} failed", flush=True)
                accepted &= {r["config_id"] for r in checked if r["status"] == "compiled"}
                print(f"{w.name}: {len(accepted)} remain valid across checked workloads", flush=True)
    result = dict(
        target=TARGET,
        gpu_execution=False,
        correctness_validated=False,
        workloads=[w.to_dict() for w in workloads],
        candidate_count=len(configs),
        configs=[c for c in configs if config_id(c) in accepted],
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    write_json(args.output / "summary.json", result)
    print(f"{args.family}: {len(accepted)}/{len(configs)} compile for all {len(workloads)} final workloads", flush=True)
    if args.publish:
        from experiments.utils.compiled_pool import publish_compiled_pool

        publish_compiled_pool(args.family, args.output, configs, original_configs(args.family, workloads[0]))


if __name__ == "__main__":
    main()

"""Controlled A100 GEMM service and register-allocation diagnostics.

This command charges all compilation and measurement. Its compiler observations
are diagnostic outputs, never features for pre-compilation selection.
"""

import argparse
import json
import time
import random
import statistics
from pathlib import Path
from dataclasses import asdict
import torch
import tilelang
from experiments.common.kernels import make_case
from experiments.common.spec import Workload, TARGETS
from tilelang.tiletune import analyze_prim_func, query_device_limits, load_device_profile
from tilelang.contrib.cuda_resource_info import CUDA_RESOURCE_CAPTURE_CONFIG_KEY


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shapes = [
        (512, 512, 256),
        (512, 512, 4096),
        (1280, 1280, 1280),
        (2048, 512, 1280),
        (512, 2048, 1280),
        (4096, 4096, 256),
        (4096, 4096, 4096),
    ]
    configs = [
        dict(block_m=128, block_n=n, block_k=k, stages=s, threads=t) for n, k, t in [(64, 64, 128), (128, 32, 256)] for s in [0, 1, 2, 3]
    ]
    configs += [dict(block_m=128, block_n=256, block_k=16, stages=s, threads=128) for s in [0, 2]]
    configs += [dict(block_m=256, block_n=256, block_k=16, stages=0, threads=128)]
    (out / "plan.json").write_text(
        json.dumps(
            dict(
                shapes=shapes,
                configs=configs,
                rounds=7,
                note="Development diagnostic; all compile and measurement costs charged here. No counters enter selection.",
            ),
            indent=2,
        )
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    rates = load_device_profile(args.profile, input_dtype="float16", memory_regime="streaming")
    limits = query_device_limits(TARGETS["ampere"])
    with (out / "outcomes.jsonl").open("w") as log:
        for shape in shapes:
            case = make_case(Workload("diagnostic", "gemm", dict(zip(("m", "n", "k"), shape))))
            inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
            expected = case.reference(*inputs)
            kernels = []
            rows = []
            for i, c in enumerate(configs):
                start = time.perf_counter()
                try:
                    f = case.build(**c)
                    r = analyze_prim_func(
                        f,
                        dict(performance_model=rates, max_local_bytes=None, max_spill_bytes=None),
                        target=TARGETS["ampere"],
                        device_limits=limits,
                        pass_configs=case.pass_configs,
                    )
                    analyzed = time.perf_counter()
                    kernel = tilelang.compile(
                        f,
                        target=TARGETS["ampere"],
                        out_idx=case.out_idx,
                        execution_backend="tvm_ffi",
                        pass_configs={**case.pass_configs, CUDA_RESOURCE_CAPTURE_CONFIG_KEY: True},
                    )
                    actual = kernel(*inputs)
                    case.check(
                        actual if isinstance(actual, (list, tuple)) else [actual],
                        expected if isinstance(expected, (list, tuple)) else [expected],
                    )
                    row = dict(
                        shape=shape,
                        index=i,
                        config=c,
                        status="correct",
                        analysis_seconds=analyzed - start,
                        compile_check_seconds=time.perf_counter() - analyzed,
                        score=r["tile_cost"]["score"],
                        wave=r["modules"]["ranking"].get("wave_timing"),
                        occupancy=r["modules"]["waves"],
                        register_demand=r["pressure"]["register_demand"],
                        resources={k: asdict(v) for k, v in kernel.resource_usage.items()},
                        samples_ms=[],
                    )
                    kernels.append((kernel.get_profiler(), row))
                except Exception as e:
                    row = dict(shape=shape, index=i, config=c, status="failed", error=str(e), seconds=time.perf_counter() - start)
                rows.append(row)
            rng = random.Random(123)
            start = time.perf_counter()
            for _ in range(7):
                rng.shuffle(kernels)
                for profiler, row in kernels:
                    row["samples_ms"].append(profiler.do_bench(input_tensors=inputs, backend="event", warmup=5, rep=20))
            duration = time.perf_counter() - start
            for row in rows:
                if row["status"] == "correct":
                    row.update(median_ms=statistics.median(row["samples_ms"]), shape_measurement_seconds=duration)
                log.write(json.dumps(row, default=str) + "\n")
                log.flush()
            print(shape, [(r["index"], r.get("median_ms"), r["status"]) for r in rows], flush=True)
            del kernels, inputs, expected
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

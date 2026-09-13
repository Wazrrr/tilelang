"""Rank the real FP8 example using a reusable device profile and one reference.

Run from the repository root. The default workflow benchmarks one config; it
analyzes all 288 supplied configs without compiling them. Use --validate-all
only to evaluate ranking quality against exhaustive tuning.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.tiletune import analyze_prim_func, anchor_latency, profile_device, rank_records
from tilelang.tiletune.config import ANALYSIS_VERSION
from tilelang.tiletune import query_device_limits
from examples.gemm_fp8.example_tilelang_gemm_fp8 import matmul, calc_diff


TARGET = {"kind": "cuda", "arch": "sm_90a"}


def get_configs():
    keys = ("block_M", "block_N", "block_K", "num_stages", "threads", "enable_rasteration")
    values = itertools.product([64, 128, 256], [64, 128, 256], [32, 64], [0, 1, 2, 3], [128, 256], [True, False])
    return [dict(zip(keys, values)) for values in values]


def make_gemm(M, N, K, dtype):
    # Elaborate the actual eager-JIT example, including normal constant binding.
    def gemm(block_M, block_N, block_K, num_stages, threads, enable_rasteration):
        return matmul.get_tir(
            M=M,
            N=N,
            K=K,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            dtype=dtype,
            num_stages=num_stages,
            threads=threads,
            enable_rasteration=enable_rasteration,
        )

    return gemm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float8_e4m3fn", "float8_e5m2"], default="float8_e4m3fn")
    parser.add_argument("--device-profile", type=Path, default=Path("fp8_device_profile.json"))
    parser.add_argument("--output", type=Path, default=Path("fp8_tiletune_report.json"))
    parser.add_argument(
        "--reference-latency-ms", type=float, help="Reuse an already measured default-config latency; run no reference benchmark"
    )
    parser.add_argument("--validate-all", action="store_true", help="Benchmark all 288 configs to evaluate the frozen ranking")
    args = parser.parse_args()
    started = time.perf_counter()
    profile = profile_device(input_dtype=args.dtype, cache_path=args.device_profile)
    profile_seconds = time.perf_counter() - started
    configs = get_configs()
    factory = make_gemm(args.size, args.size, args.size, args.dtype)
    limits = query_device_limits(TARGET)
    settings = dict(mode="report_only", ranking_metric="pipeline_time", performance_model=profile)
    records, reference_func = [], None
    reference_config = dict(block_M=128, block_N=128, block_K=64, num_stages=3, threads=128, enable_rasteration=False)
    started = time.perf_counter()
    for index, config in enumerate(configs):
        func = factory(**config)
        records.append(dict(index=index, config=config, **analyze_prim_func(func, settings, target=TARGET, device_limits=limits)))
        if config == reference_config:
            reference_func, reference_index = func, index
    analysis_seconds = time.perf_counter() - started
    ranking = rank_records(records)
    report = dict(
        analysis_version=ANALYSIS_VERSION,
        dtype=args.dtype,
        size=args.size,
        config_count=len(configs),
        profile=profile,
        profile_seconds=profile_seconds,
        analysis_seconds=analysis_seconds,
        analysis_source_sha256={
            p.relative_to("tilelang/tiletune").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("tilelang/tiletune").rglob("*.py")
        },
        records=records,
        ranking=ranking,
        reference_config=reference_config,
        reference_index=reference_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze the profile, predictions and their source hashes before target timings.
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    inputs = None
    if args.reference_latency_ms is None or args.validate_all:
        torch.manual_seed(123)
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = T.dtype(args.dtype).as_torch()
        inputs = [(torch.rand((args.size, args.size), device="cuda", dtype=torch.float16) - 0.5).to(dtype) for _ in range(2)]
    latency = args.reference_latency_ms
    if latency is None:
        kernel = tilelang.compile(reference_func, target=TARGET, execution_backend="tvm_ffi")
        actual = kernel(*inputs)
        reference = (inputs[0].float() @ inputs[1].float().T).to(actual.dtype)
        difference = calc_diff(actual, reference).item()
        if difference >= 1e-3:
            raise AssertionError(f"FP8 example correctness failed: calc_diff={difference}")
        latency = kernel.get_profiler().do_bench(input_tensors=inputs, n_warmup=10, n_repeat=100, backend="cudagraph")
        report["reference_accuracy"] = dict(calc_diff=difference, threshold=1e-3)
    anchored = anchor_latency(profile, records[reference_index], latency)
    report.update(
        reference_latency_ms=latency,
        anchored_profile=anchored,
        target_configs_benchmarked=0 if args.reference_latency_ms is not None else 1,
    )
    if args.validate_all:

        def check(actuals, refs):
            difference = calc_diff(actuals[0], refs[0]).item()
            assert difference < 1e-3, f"FP8 example correctness failed: calc_diff={difference}"

        tuner = (
            AutoTuner(factory, configs)
            .set_compile_args(target=TARGET, execution_backend="tvm_ffi")
            .set_tiletune_args(True, **settings, report_path=str(args.output.with_suffix(".tuning.json")))
            .set_profile_args(
                ref_prog=lambda a, b: (a.float() @ b.float().T).to(a.dtype),
                supply_prog=lambda params: inputs,
                backend="cudagraph",
                manual_check_prog=check,
                cache_input_tensors=True,
            )
        )
        result = tuner.run(warmup=10, rep=100, early_stop=False)
        report.update(winner_config=result.config, winner_latency_ms=result.latency, target_configs_benchmarked=len(configs))
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            dict(
                reference_latency_ms=latency,
                latency_scale=anchored["latency_scale"],
                top_configs=[dict(r, config=configs[r["index"]]) for r in ranking[:8]],
                report=str(args.output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

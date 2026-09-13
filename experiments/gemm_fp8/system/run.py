"""Measure FP8 GEMM autotuning with each system optimization, or all variants."""

import argparse
import json
import subprocess
import sys
import time

from experiments._common import (
    add_gemm_arguments,
    add_run_arguments,
    device_info,
    prepare_run,
    select_configs,
    source_hashes,
    write_json,
)


# Values are (compile/benchmark overlap, grouped compilation, multi-GPU benchmarking).
VARIANTS = {
    "baseline": (False, False, False),
    "pipeline": (True, False, False),
    "grouped": (False, True, False),
    "multi_gpu": (False, False, True),
    "combined": (True, True, True),
}


def run_all(args):
    """Use a fresh process for each variant, with the same grid and inputs."""
    rows = []
    for variant in VARIANTS:
        output = args.output / variant
        command = [
            sys.executable,
            "-m",
            "experiments.gemm_fp8.system.run",
            "--variant",
            variant,
            "--output",
            str(args.output),
            "--run-name",
            variant,
        ]
        for name in ("m", "n", "k", "dtype", "workers", "warmup", "rep", "timeout", "seed", "group_size", "backend"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        command += ["--benchmark-devices", *map(str, args.benchmark_devices)]
        if args.config_indices is not None:
            command += ["--config-indices", *map(str, args.config_indices)]
        subprocess.run(command, check=True)
        rows.append(json.loads((output / "summary.json").read_text()))
    baseline = rows[0]["tuning_seconds"]
    print("\nVariant       Tuning (s)  Speedup  Winner (ms)")
    for row in rows:
        row["tuning_speedup_vs_baseline"] = baseline / row["tuning_seconds"]
        print(
            f"{row['variant']:12} {row['tuning_seconds']:10.3f} {row['tuning_speedup_vs_baseline']:7.2f}x {row['winner_latency_ms']:12.6f}"
        )
    write_json(args.output / "comparison.json", rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, "experiments/results/gemm_fp8/system")
    add_gemm_arguments(parser)
    parser.add_argument("--dtype", choices=["float8_e4m3fn", "float8_e5m2"], default="float8_e4m3fn")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="baseline")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--benchmark-devices", type=int, nargs="+", default=[0], help="Logical ordinals within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--backend", choices=["event", "cudagraph"], default="event")
    args = parser.parse_args(argv)
    if args.group_size < 2:
        parser.error("--group-size must be at least 2")
    return args


def main():
    args = parse_args()
    prepare_run(args)

    import torch

    devices = args.benchmark_devices
    if len(set(devices)) != len(devices) or any(d < 0 or d >= torch.cuda.device_count() for d in devices):
        raise ValueError("--benchmark-devices must name distinct visible CUDA devices")
    if args.variant in ("multi_gpu", "combined", "all") and len(devices) < 2:
        raise ValueError("this variant requires at least two --benchmark-devices")
    if args.variant == "all":
        run_all(args)
        return

    from tilelang.autotuner import AutoTuner, set_autotune_inputs
    from experiments.gemm_fp8.kernel import check_accuracy, get_configs, make_inputs, make_kernel, reference
    from tilelang.tiletune.profiling.device_profile import current_target

    torch.cuda.set_device(devices[0])
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline, grouped, multi_gpu = VARIANTS[args.variant]
    grid = get_configs()
    indices, configs = select_configs(grid, args.config_indices)
    inputs = make_inputs(args.m, args.n, args.k, args.dtype, args.seed)
    active_devices = devices if multi_gpu else devices[:1]
    references = {}
    for device in active_devices:
        values = [tensor.to(f"cuda:{device}") for tensor in inputs]
        references[device] = reference(*values)
        torch.cuda.synchronize(device)
    target = current_target()
    with set_autotune_inputs(inputs):
        tuner = (
            AutoTuner(make_kernel(args.m, args.n, args.k, args.dtype), configs)
            .set_compile_args(out_idx=[2], target=target, execution_backend="tvm_ffi")
            .set_profile_args(
                ref_prog=lambda *values: references[values[0].device.index],
                manual_check_prog=check_accuracy,
                backend=args.backend,
                cache_input_tensors=False,
            )
            .set_benchmark_report_path(str(args.output / "benchmarks.tsv"))
        )
    settings = dict(
        variant=args.variant,
        arguments=vars(args),
        devices=device_info(devices if multi_gpu else devices[:1]),
        original_grid_size=len(grid),
        original_indices=indices,
        configs=configs,
        target=target,
        source_sha256=source_hashes("examples/gemm_fp8/example_tilelang_gemm_fp8.py"),
        use_pipeline=pipeline,
        enable_grouped_compile=grouped,
        benchmark_multi_gpu=multi_gpu,
        cold_kernel_cache=True,
        cold_autotune_cache=True,
    )
    write_json(args.output / "experiment.json", settings)
    print(f"{args.variant}: tuning {len(configs)}/{len(grid)} configurations", flush=True)
    started = time.perf_counter()
    result = tuner.run(
        warmup=args.warmup,
        rep=args.rep,
        timeout=args.timeout,
        early_stop=False,
        use_pipeline=pipeline,
        enable_grouped_compile=grouped,
        group_compile_size=args.group_size,
        benchmark_multi_gpu=multi_gpu,
        benchmark_devices=devices if multi_gpu else devices[:1],
    )
    summary = dict(
        variant=args.variant,
        tuning_seconds=time.perf_counter() - started,
        config_count=len(configs),
        winner_config=result.config,
        winner_latency_ms=result.latency,
    )
    write_json(args.output / "summary.json", summary)
    print(f"Tuning: {summary['tuning_seconds']:.3f} s; measured winner: {result.latency:.6f} ms")
    print(f"Winner config: {result.config}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()

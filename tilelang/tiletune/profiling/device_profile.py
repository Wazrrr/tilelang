"""Reusable, dtype-specific device measurements and one-sample latency anchors.

No candidate-grid fitting. Explicit profiling measures fixed primitive kernels;
loading and applying a cached profile never executes a kernel.
"""

import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

PROFILE_VERSION = 4


def _signature(input_dtype, accum_dtype, instruction="cuda.wgmma"):
    return {"instruction": instruction, "a_dtype": str(input_dtype), "b_dtype": str(input_dtype), "accum_dtype": str(accum_dtype)}


def current_target():
    """Return the supported target for the current visible CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise ValueError("device profiling requires a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (8, 0):
        return {"kind": "cuda", "arch": "sm_80"}
    if major == 9:
        return {"kind": "cuda", "arch": "sm_90a"}
    raise ValueError("device profiling supports Ampere sm_80 (A100) and Hopper sm_90a GPUs")


def _instruction(identity):
    return "cuda.mma" if identity["target_arch"] == "sm_80" else "cuda.wgmma"


def _identity():
    import torch
    import tilelang
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.contrib.nvcc import find_cuda_path

    target = current_target()
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    source = Path(__file__).with_name("device_probes.py")
    return {
        "profile_version": PROFILE_VERSION,
        "target_arch": target["arch"],
        "device_name": prop.name,
        "sm_count": prop.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "tilelang_version": tilelang.__version__,
        "probe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_build": KernelCache._get_tilelang_lib_stamp(),
        "nvcc_version": subprocess.check_output([str(Path(find_cuda_path()) / "bin/nvcc"), "--version"], text=True).strip(),
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "l2_cache_bytes": prop.L2_cache_size,
    }


def load_device_profile(path, *, input_dtype, accum_dtype="float32", expected_identity=None, memory_regime="cached"):
    """Load a cached measurement without CUDA queries or kernel execution."""
    data = json.loads(Path(path).read_text())
    if memory_regime not in ("cached", "streaming"):
        raise ValueError("memory_regime must be cached or streaming")
    if data.get("identity", {}).get("profile_version") not in (2, 3, PROFILE_VERSION):
        raise ValueError("unsupported device profile version; regenerate the profile")
    if expected_identity is not None and data["identity"] != expected_identity:
        raise ValueError("device/profile fingerprint mismatch; use a separate cache path or refresh explicitly")
    signature = _signature(input_dtype, accum_dtype, _instruction(data["identity"]))
    key = json.dumps(signature, sort_keys=True)
    if key not in data["gemm_models"]:
        raise ValueError(f"device profile has no measurements for {signature}")
    model = data["gemm_models"][key]
    rates = {**data["common"]["rates"], **model["rates"]}
    streaming_rate = rates.pop("dram_bytes_per_cycle", None)
    if memory_regime == "streaming":
        if streaming_rate is None:
            raise ValueError("device profile has no streaming memory measurement")
        rates["global_bytes_per_cycle"] = streaming_rate
    if data["common"].get("consumer_rates"):
        rates["consumer_rates"] = data["common"]["consumer_rates"]
    return {
        **rates,
        "gemm_signature": signature,
        "profile_target": data["identity"]["target_arch"],
        "profile_id": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "reference_clock_mhz": data["common"]["clock_mhz"],
        "memory_regime": memory_regime,
        "reduction_dtype": "float32",
    }


def anchor_latency(performance_model, analysis, measured_latency_ms):
    """Calibrate one positive global scale; it cannot change config ordering.

    The analysis must use the supplied profile. A single elapsed time cannot
    identify separate memory, compute and synchronization rates.
    """
    if not math.isfinite(measured_latency_ms) or measured_latency_ms <= 0:
        raise ValueError("reference latency must be finite and positive")
    clock = performance_model.get("reference_clock_mhz")
    score = analysis.get("tile_cost", {}).get("score")
    used = analysis.get("modules", {}).get("pipeline_overlap", {}).get("performance_model")
    if (
        not clock
        or score is None
        or score <= 0
        or used != performance_model
        or analysis["tile_cost"].get("ranking_metric") != "pipeline_time"
    ):
        raise ValueError("anchor requires a scored pipeline analysis using this exact profile and its reference clock")
    result = copy.deepcopy(performance_model)
    result["latency_scale"] = performance_model.get("latency_scale", 1) * measured_latency_ms * clock * 1000 / score
    return result


def profile_device(*, input_dtype="float16", accum_dtype="float32", cache_path=None, refresh=False, memory_regime="cached"):
    """Measure missing primitive rates once and reuse them across kernels.

    A new device uses fixed common probes plus dtype-specific MMA/WGMMA probes.
    Adding a dtype reuses common measurements. No kernel/config supplied by the
    caller is benchmarked here. Existing incompatible files require refresh.
    """
    import fcntl
    from tilelang import env

    if memory_regime not in ("cached", "streaming"):
        raise ValueError("memory_regime must be cached or streaming")
    identity = _identity()
    if str(input_dtype) not in ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2") or str(accum_dtype) != "float32":
        raise ValueError("device probes support FP16/BF16/FP8 inputs with FP32 accumulation")
    if identity["target_arch"] == "sm_80" and str(input_dtype).startswith("float8"):
        raise ValueError("A100 has no FP8 tensor-core instructions; use FP16 or BF16")
    if cache_path is None:
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cache_path = Path(env.TILELANG_CACHE_DIR) / "tiletune_profiles" / f"{key}.json"
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = None
        if path.exists() and not refresh:
            data = json.loads(path.read_text())
            if data.get("identity") != identity:
                raise ValueError("device/profile fingerprint mismatch; use a separate cache path or refresh explicitly")
        if data is None:
            data = {"identity": identity, "common": _measure_common(identity), "gemm_models": {}}
        signature = _signature(input_dtype, accum_dtype, _instruction(identity))
        key = json.dumps(signature, sort_keys=True)
        if key not in data["gemm_models"]:
            data["gemm_models"][key] = _measure_gemm(identity, data["common"]["clock_mhz"], input_dtype, accum_dtype)
            data["updated_at_unix"] = time.time()
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            os.replace(temporary, path)
    return load_device_profile(
        path, input_dtype=input_dtype, accum_dtype=accum_dtype, expected_identity=identity, memory_regime=memory_regime
    )


def _clock_mhz():
    import torch

    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(prop.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    text = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], text=True)
    value = float(text.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("unable to read a positive SM reference clock")
    return value


def _benchmark(func, inputs, out_idx, check=None, required_source=None, *, target=None):
    import torch
    import tilelang

    kernel = tilelang.compile(func, target=target or current_target(), out_idx=out_idx, execution_backend="tvm_ffi")
    source = kernel.get_kernel_source()
    if required_source is not None and required_source not in source:
        raise RuntimeError(f"primitive did not compile to its required instruction: {required_source}")
    output = kernel(*inputs)
    if not torch.isfinite(output.float()).all():
        raise RuntimeError("non-finite primitive probe output")
    if check is not None:
        check(output)
    profiler = kernel.get_profiler()
    samples = [profiler.do_bench(input_tensors=inputs, n_warmup=10, n_repeat=50, backend="cudagraph") for _ in range(3)]
    return (
        statistics.median(samples),
        {
            "latency_samples_ms": samples,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "required_instruction_verified": required_source,
        },
        output,
    )


def _slope(short_ms, long_ms, iterations, clock):
    value = (long_ms - short_ms) * 1000 * clock / iterations
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("nonpositive primitive timing slope; rerun profiling on an idle GPU")
    return value


def _measure_common(identity):
    import torch
    from . import device_probes as probes

    torch.cuda.synchronize()
    # Wake the device before reading its reference clock.
    torch.ones((1024, 1024), device="cuda").sum().item()
    clock, sms = _clock_mhz(), identity["sm_count"]
    rates, evidence = {}, {}
    cached_elements = (identity["l2_cache_bytes"] // 16 // 4096) * 4096
    for label, elements in (("cached", cached_elements), ("streaming", 64 * 1024 * 1024)):
        measurements = []
        for size in (elements // 2, elements):
            a = torch.rand((size,), device="cuda")
            ms, obs, _ = _benchmark(probes.memory_copy(size), [a], [1], lambda out, a=a: torch.testing.assert_close(out, a))
            obs.update(bytes=size * 8, footprint_bytes=size * 8)
            measurements.append((ms, obs))
            del a
        cycles = _slope(measurements[0][0], measurements[1][0], 1, clock)
        evidence[label] = {"measurements": [m[1] for m in measurements], "method": "paired footprint sizes subtract fixed launch costs"}
        rates["global_bytes_per_cycle" if label == "cached" else "dram_bytes_per_cycle"] = elements * 4 / (cycles * sms)
    # Paired loop counts subtract fixed launch/setup costs; these are effective
    # primitive rates including instruction dependencies, not nominal peaks.
    blocks = sms * 4
    counts = {
        0: ("elementwise_ops_per_cycle", 128 * 8 * 2),
        1: ("shared_bytes_per_cycle", 128 * 8 * 8),
        2: ("exp_ops_per_cycle", 128 * 8),
        3: ("reduction_ops_per_cycle", 4 * 8 * 31),
    }
    for kind, (name, work) in counts.items():
        measurements = [_benchmark(probes.primitive(kind, n, blocks), [], [0]) for n in (256, 512)]
        cycles = _slope(measurements[0][0], measurements[1][0], 256, clock)
        rates[name] = work * blocks / (cycles * sms)
        evidence[name] = {"work_per_block_iteration": work, "iterations": [256, 512], "measurements": [m[1] for m in measurements]}
    # Separate physical instruction work from logical reduced-element counts.
    for kind, name in enumerate(
        (
            "reduction_local_sum_per_cycle",
            "reduction_local_max_per_cycle",
            "reduction_shuffle_sum_per_cycle",
            "reduction_shuffle_max_per_cycle",
        )
    ):
        measurements = [_benchmark(probes.reduction_primitive(kind, n, blocks), [], [0]) for n in (256, 512)]
        cycles = _slope(measurements[0][0], measurements[1][0], 256, clock)
        work = 128 * 8
        rates[name] = work * blocks / (cycles * sms)
        evidence[name] = {
            "work_per_block_iteration": work,
            "iterations": [256, 512],
            "measurements": [m[1] for m in measurements],
            "method": "physical FP32 local pair" if kind < 2 else "physical lane shuffle/combine pair; normalization included for sum",
        }
    _, obs, output = _benchmark(probes.primitive(4, 512, 1), [], [0])
    rates["barrier_cycles"] = output.float().median().item() / 512
    evidence["barrier_cycles"] = {**obs, "method": "device clock64 over 512 volatile bar.sync instructions, one CTA"}
    a = torch.rand((sms, 32, 32), device="cuda")
    hopper = identity["target_arch"] == "sm_90a"
    measurements = [
        _benchmark(
            probes.tile_copy_roundtrip(n, sms, "tma" if hopper else "sync"),
            [a],
            [1],
            lambda out, n=n: torch.testing.assert_close(out, a * n, rtol=0.001, atol=0.001),
            required_source="tl::tma_load" if hopper else None,
        )
        for n in (128, 256)
    ]
    roundtrip = _slope(measurements[0][0], measurements[1][0], 128, clock)
    rates["copy_latency_cycles"] = max(0.0, roundtrip - 4096 / rates["global_bytes_per_cycle"] - rates["barrier_cycles"])
    evidence["copy_latency_cycles"] = {
        "roundtrip_cycles": roundtrip,
        "bytes": 4096,
        "measurements": [m[1] for m in measurements],
        "method": ("TMA" if hopper else "synchronous")
        + " tile copy + dependent consumer; subtract measured byte service and barrier; residual includes consumer overhead",
    }
    return {
        "rates": rates,
        **_measure_consumer_rates(clock),
        "clock_mhz": clock,
        "evidence": evidence,
        "memory_regime": "cached logical tile service; streaming rate measured separately",
        "note": "Fixed primitives independent of candidate kernels. Rates include probe-specific overhead and may not predict absolute runtime.",
    }


def _measure_gemm(identity, clock, input_dtype, accum_dtype):
    import torch
    import tilelang.language as T
    from .device_probes import tensor_core

    dtype = T.dtype(input_dtype).as_torch()
    a = torch.full((64, 128), 0.125, dtype=torch.float32, device="cuda").to(dtype)
    b = torch.full((128, 128), 0.125, dtype=torch.float32, device="cuda").to(dtype)
    rates, evidence = {}, {}
    hopper = _instruction(identity) == "cuda.wgmma"
    probes = [("gemm_flops_per_cycle", identity["sm_count"] * 4, 256)]
    if hopper:
        probes.append(("wgmma_flops_per_cycle_per_warpgroup", 1, 128))
    for name, blocks, threads in probes:
        measurements = [
            _benchmark(
                tensor_core(input_dtype, accum_dtype, n, blocks, threads),
                [a, b],
                [2],
                lambda out: torch.testing.assert_close(out, torch.zeros_like(out)),
                required_source="tl::wgmma_ss" if hopper else "tl::mma_sync",
            )
            for n in (128, 256)
        ]
        cycles = _slope(measurements[0][0], measurements[1][0], 128, clock)
        rates[name] = 2 * 64 * 128 * 128 * blocks / (cycles * (identity["sm_count"] if blocks > 1 else 1))
        evidence[name] = {
            "tile": [64, 128, 128],
            "blocks": blocks,
            "threads": threads,
            "iterations": [128, 256],
            "measurements": [m[1] for m in measurements],
            "method": "shared-resident "
            + _instruction(identity)
            + " loop with bounded accumulator dependency (negation each iteration); no global transfers inside measured loop",
        }
    return {"rates": rates, "evidence": evidence}


def _measure_consumer_rates(clock):
    """Single-CTA service ceilings, independently of aggregate SM throughput.

    Fixed eight-chain primitive kernels are swept across launch thread counts.
    No candidate tile, kernel family, or candidate latency is supplied here.
    """
    from . import device_probes as probes

    rates, evidence = {}, {}
    for threads in (32, 64, 128, 256, 512):
        row, observations = {}, {}
        kinds = [
            (probes.primitive, 0, "elementwise_ops_per_cycle", 2),
            (probes.primitive, 2, "exp_ops_per_cycle", 1),
            *[
                (probes.reduction_primitive, kind, name, 1)
                for kind, name in enumerate(
                    (
                        "reduction_local_sum_per_cycle",
                        "reduction_local_max_per_cycle",
                        "reduction_shuffle_sum_per_cycle",
                        "reduction_shuffle_max_per_cycle",
                    )
                )
            ],
        ]
        for factory, kind, name, factor in kinds:
            measurements = [_benchmark(factory(kind, n, 1, threads), [], [0]) for n in (256, 512)]
            cycles = _slope(measurements[0][0], measurements[1][0], 256, clock)
            row[name] = threads * 8 * factor / cycles
            observations[name] = dict(
                threads=threads,
                blocks=1,
                independent_chains=8,
                iterations=[256, 512],
                work_per_iteration=threads * 8 * factor,
                measurements=[m[1] for m in measurements],
            )
        rates[str(threads)], evidence[str(threads)] = row, observations
    return dict(
        consumer_rates=rates,
        consumer_rate_evidence=evidence,
        consumer_rate_method="single-CTA fixed primitive service ceilings; aggregate SM ceilings applied separately",
    )

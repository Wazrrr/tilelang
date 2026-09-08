"""Frozen, exhaustive cross-workload cost-model experiments; no candidate fitting."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark.autotune.benchmark_new_carver_attention import AttentionTuner, write
from benchmark.autotune.benchmark_new_carver_gemm import full_configs, snapshot
from benchmark.autotune.new_carver_experiment_utils import (
    factory as attention_factory,
    inputs_and_reference as attention_inputs,
    evaluate_ranking,
    traffic_ranking,
)
from examples.flash_attention.example_mha_new_carver import PASS_CONFIGS, check_accuracy, get_configs
from tilelang.new_carver import CarverConfig, analyze_prim_func, load_device_profile, rank_records
from tilelang.new_carver.config import ANALYSIS_VERSION
from tilelang.new_carver.device_profile import _identity, current_target, profile_device
import tilelang.language as T

DEVELOPMENT = [
    dict(name="gemm_square", family="gemm", m=1536, n=1536, k=1536, dtype="float16", trans_a=False, trans_b=False),
    dict(name="gemm_tall", family="gemm", m=8192, n=512, k=2048, dtype="float16", trans_a=False, trans_b=False),
    dict(name="gemm_short_k", family="gemm", m=4096, n=4096, k=256, dtype="float16", trans_a=False, trans_b=False),
    dict(name="gemm_transpose", family="gemm", m=2048, n=4096, k=3072, dtype="float16", trans_a=False, trans_b=True),
]
# Declared before development measurements; do not inspect their timings until model freeze.
HOLDOUT = [
    dict(name="gemm_rect", family="gemm", m=3072, n=6144, k=1536, dtype="float16", trans_a=False, trans_b=False),
    dict(name="gemm_wide_nt", family="gemm", m=768, n=8192, k=4096, dtype="float16", trans_a=False, trans_b=True),
    dict(name="gemm_bf16_tn", family="gemm", m=6144, n=1536, k=512, dtype="bfloat16", trans_a=True, trans_b=False),
    dict(name="gemm_fp8", family="gemm", m=4096, n=2048, k=4096, dtype="float8_e4m3fn", trans_a=False, trans_b=False),
    *[
        dict(
            name=name + ("_causal" if causal else "_noncausal"),
            family="attention",
            layout=layout,
            batch=batch,
            heads=heads,
            query_length=sq,
            kv_length=sk,
            dim=dim,
            dtype="float16",
            causal=causal,
        )
        for name, layout, batch, heads, sq, sk, dim in [
            ("attn_d64", "BSHD", 2, 12, 3072, 3072, 64),
            ("attn_d256", "BSHD", 1, 8, 2048, 2048, 256),
            ("attn_rect", "BHSD", 2, 4, 1536, 3072, 128),
        ]
        for causal in (False, True)
    ],
]


BASELINE = [
    dict(name="gemm_4096", family="gemm", m=4096, n=4096, k=4096, dtype="float16", trans_a=False, trans_b=False),
    *[
        dict(
            name="attn_4096_" + ("causal" if causal else "noncausal"),
            family="attention",
            layout="BSHD",
            batch=1,
            heads=16,
            query_length=4096,
            kv_length=4096,
            dim=128,
            dtype="float16",
            causal=causal,
        )
        for causal in (False, True)
    ],
]
SPLITS = {"baseline": BASELINE, "development": DEVELOPMENT, "holdout": HOLDOUT, "all": BASELINE + DEVELOPMENT + HOLDOUT}
REPO_ROOT = Path(__file__).resolve().parents[2]


def unsupported_reason(workload, target):
    if target["arch"] == "sm_80" and workload["dtype"].startswith("float8"):
        return "A100 has no FP8 tensor-core instructions"
    return None


def measure_profile(path, refresh=False):
    target = current_target()
    dtypes = ["float16", "bfloat16"]
    if target["arch"] == "sm_90a":
        dtypes.append("float8_e4m3fn")
    for index, dtype in enumerate(dtypes):
        print(f"Measuring/reusing fixed {dtype} primitives for {target['arch']}", flush=True)
        profile_device(input_dtype=dtype, cache_path=path, refresh=refresh and index == 0, memory_regime="streaming")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configs_for(w):
    return full_configs() if w["family"] == "gemm" else get_configs()


def factory(w):
    if w["family"] == "attention":
        return attention_factory(w, w["causal"])

    m, n, k, dtype, ta, tb = (w[x] for x in ("m", "n", "k", "dtype", "trans_a", "trans_b"))

    def gemm(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
        @T.prim_func
        def main(
            A: T.Tensor((k, m) if ta else (m, k), dtype), B: T.Tensor((n, k) if tb else (k, n), dtype), C: T.Tensor((m, n), "float32")
        ):
            with T.Kernel(T.ceildiv(m, block_M), T.ceildiv(n, block_N), threads=thread_num) as (bx, by):
                a = T.alloc_shared((block_K, block_M) if ta else (block_M, block_K), dtype)
                b = T.alloc_shared((block_N, block_K) if tb else (block_K, block_N), dtype)
                c = T.alloc_fragment((block_M, block_N), "float32")
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(c)
                for ki in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                    if ta:
                        T.copy(A[ki * block_K, bx * block_M], a)
                    else:
                        T.copy(A[bx * block_M, ki * block_K], a)
                    if tb:
                        T.copy(B[by * block_N, ki * block_K], b)
                    else:
                        T.copy(B[ki * block_K, by * block_N], b)
                    T.gemm(a, b, c, transpose_A=ta, transpose_B=tb)
                T.copy(c, C[bx * block_M, by * block_N])

        return main

    return gemm


def freeze(root, profile, split, *, mode="report_only", group_size=1):
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.new_carver.cost import query_device_limits

    if (root / "freeze.json").exists():
        raise ValueError("frozen experiment already exists")
    target = current_target()
    identity = _identity()
    for w in SPLITS[split]:
        if not unsupported_reason(w, target):
            load_device_profile(profile, input_dtype=w["dtype"], expected_identity=identity, memory_regime="streaming")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(profile, root / "device.json")
    paths = [
        *Path("tilelang/new_carver").rglob("*.py"),
        Path(__file__),
        Path("benchmark/autotune/benchmark_new_carver_attention.py"),
        Path("benchmark/autotune/benchmark_new_carver_gemm.py"),
        Path("benchmark/autotune/new_carver_experiment_utils.py"),
        *Path("examples/flash_attention").glob("example_mha*.py"),
        *Path("tilelang/autotuner").glob("*.py"),
        Path("tilelang/engine/lower.py"),
        Path("tilelang/profiler/bench.py"),
        Path("tilelang/utils/autotune_timing.py"),
        Path("tilelang/jit/adapter/tvm_ffi.py"),
        Path("tilelang/contrib/cuda_resource_info.py"),
        Path("src/op/operator.cc"),
        Path("src/cuda/op/copy_analysis.cc"),
        Path("src/cuda/transform/producer_consumer_ws.cc"),
    ]
    hashes = {}
    for path in paths:
        relative = path.resolve().relative_to(Path.cwd())
        dest = root / "sources" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        hashes[str(relative)] = digest(path)
    write(
        root / "freeze.json",
        dict(
            frozen_at=datetime.now(timezone.utc).isoformat(),
            analysis_version=ANALYSIS_VERSION,
            split=split,
            workloads=SPLITS[split],
            target=target,
            group_size=group_size,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], text=True),
            reserved_holdout=HOLDOUT,
            profile_sha256=digest(root / "device.json"),
            source_sha256=hashes,
            native_build=KernelCache._get_tilelang_lib_stamp(),
            device_limits=query_device_limits(target),
            mode=mode,
            ranking_metric="pipeline_time",
            memory_regime="streaming",
            gemm_spill_bytes=0,
            attention_spill_bytes=None,
            attention_register_margin=32,
            warmup=5,
            repeats=30,
            seed=123,
            compile_workers=4,
            note="No kernel/config-specific fitted parameters. All supplied configs run; failures and unknowns count.",
        ),
    )


def verify(root):
    from tilelang.cache.kernel_cache import KernelCache

    p = json.loads((root / "freeze.json").read_text())
    assert p["analysis_version"] == ANALYSIS_VERSION
    assert p["profile_sha256"] == digest(root / "device.json")
    for path, expected in p["source_sha256"].items():
        assert digest(path) == expected, path
    assert p["native_build"] == KernelCache._get_tilelang_lib_stamp()
    return p


def settings(w, profile):
    attention = w["family"] == "attention"
    return dict(
        mode="report_only",
        ranking_metric="pipeline_time",
        performance_model=profile,
        max_spill_bytes=None if attention else 0,
        max_local_bytes=None if attention else 0,
        attention_spill_budget_registers_per_thread=32 if attention else 0,
    )


def run(root, w, gpu=None):
    import torch

    protocol = verify(root)
    identity = _identity()
    if json.loads((root / "device.json").read_text())["identity"] != identity:
        raise ValueError("device/profile fingerprint mismatch; freeze and profile on this device type")
    output = root / w["name"]
    if (output / "summary.json").exists():
        raise ValueError("completed measurements already exist")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(output / "timings.tsv")
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = "4"
    reason = unsupported_reason(w, protocol["target"])
    if reason:
        rows = [
            dict(index=i, config=c, status="unsupported", compile_status="unsupported", benchmark_status="not_run", reason=reason)
            for i, c in enumerate(configs_for(w))
        ]
        write(
            output / "summary.json",
            dict(
                workload=w,
                status="unsupported",
                reason=reason,
                config_count=len(rows),
                configs=rows,
                protocol_sha256=digest(root / "freeze.json"),
            ),
        )
        print(f"{w['name']}: {reason} ({len(rows)} config outcomes)", flush=True)
        return
    profile = load_device_profile(root / "device.json", input_dtype=w["dtype"], expected_identity=identity, memory_regime="streaming")
    attention = w["family"] == "attention"
    before = snapshot(gpu)
    if attention:
        inputs, reference = attention_inputs(w, w["causal"])
    else:
        torch.manual_seed(123)
        torch.backends.cuda.matmul.allow_tf32 = False
        a = (torch.rand((w["m"], w["k"]), device="cuda") - 0.5).to(T.dtype(w["dtype"]).as_torch())
        b = (torch.rand((w["k"], w["n"]), device="cuda") - 0.5).to(T.dtype(w["dtype"]).as_torch())
        reference = a.float() @ b.float()
        inputs = [a.T.contiguous() if w["trans_a"] else a, b.T.contiguous() if w["trans_b"] else b]
    accuracies = []

    def check(actuals, refs):
        if attention:
            accuracies.append(check_accuracy(actuals[0], refs[0]))
        else:
            torch.testing.assert_close(actuals[0], refs[0], rtol=0.01, atol=0.01)
            accuracies.append({"max_absolute_error": (actuals[0] - refs[0]).abs().max().item()})

    configs = configs_for(w)
    tuner = (
        AttentionTuner(factory(w), configs)
        .set_compile_args(
            target=protocol["target"],
            execution_backend="tvm_ffi",
            out_idx=[3 if attention else 2],
            pass_configs=PASS_CONFIGS if attention else {"tl.enable_cuda_resource_capture": True},
        )
        .set_profile_args(
            ref_prog=lambda *args: reference,
            supply_prog=lambda params: inputs,
            manual_check_prog=check,
            backend="cudagraph",
            cache_input_tensors=True,
        )
        .set_benchmark_report_path(str(output / "benchmarks.tsv"))
    )
    tuner.set_carver_args(
        protocol["mode"] != "disabled",
        report_path=str(output / "carver.json"),
        **{**settings(w, profile), "mode": "reject" if protocol["mode"] == "reject" else "report_only"},
    )
    tuner.run_dir, tuner.record_errors = output, []
    tuner.outcomes = [
        dict(index=i, original_index=i, config=c, compile_status="pending", benchmark_status="not_run") for i, c in enumerate(configs)
    ]
    result, error, started = None, None, time.perf_counter()
    try:
        result = tuner.run(
            warmup=protocol["warmup"],
            rep=protocol["repeats"],
            timeout=30,
            early_stop=False,
            use_pipeline=False,
            enable_grouped_compile=protocol["group_size"] > 1,
            group_compile_size=protocol["group_size"],
        )
    except Exception as exc:
        error = str(exc)
    write(
        output / "summary.json",
        dict(
            workload=w,
            gpu=gpu,
            target=protocol["target"],
            mode=protocol["mode"],
            wall_seconds=time.perf_counter() - started,
            config_count=len(configs),
            configs=tuner.outcomes,
            error=error,
            record_errors=tuner.record_errors,
            correctness=accuracies,
            carver=tuner.carver_report,
            winner_config=result.config if result else None,
            winner_latency_ms=result.latency if result else None,
            gpu_before=before,
            gpu_after=snapshot(gpu),
            protocol_sha256=digest(root / "freeze.json"),
        ),
    )
    verify(root)
    print(json.dumps(dict(case=w["name"], error=error, winner=result.config if result else None)), flush=True)
    if error or tuner.record_errors:
        raise SystemExit(1)


def worker(root, gpu=None, shard=0, shards=1):
    p = verify(root)
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("require 0 <= shard < shards")
    results = []
    for w in p["workloads"][shard::shards]:
        completed_path = root / w["name"] / "summary.json"
        if completed_path.exists():
            completed = json.loads(completed_path.read_text())
            failed = bool(completed.get("error") or completed.get("record_errors"))
            results.append(dict(case=w["name"], returncode=int(failed), resumed=True))
            write(root / f"worker_shard{shard}.json", results)
            continue
        with (root / (w["name"] + ".log")).open("w") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.autotune.validate_new_carver_generalization",
                    "run",
                    "--root",
                    str(root),
                    *(["--gpu", str(gpu)] if gpu is not None else []),
                    "--case",
                    w["name"],
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results.append(dict(case=w["name"], returncode=completed.returncode))
        write(root / f"worker_shard{shard}.json", results)
        print(results[-1], flush=True)
    verify(root)
    if any(r["returncode"] for r in results):
        raise SystemExit(1)


def evaluate(root):
    p = json.loads((root / "freeze.json").read_text())
    results = []
    for w in p["workloads"]:
        s = json.loads((root / w["name"] / "summary.json").read_text())
        if s.get("status") == "unsupported":
            results.append(dict(case=w["name"], workload=w, status="unsupported", reason=s["reason"], config_count=s["config_count"]))
            continue
        if s.get("carver") is None:
            results.append(
                dict(
                    case=w["name"],
                    workload=w,
                    mode=s.get("mode"),
                    error=s.get("error"),
                    winner_config=s.get("winner_config"),
                    winner_latency_ms=s.get("winner_latency_ms"),
                    wall_seconds=s["wall_seconds"],
                    config_count=s["config_count"],
                )
            )
            continue
        records, measured = s["carver"]["configs"], s["configs"]
        assert len(records) == len(measured) == len(configs_for(w))
        assert [r["index"] for r in records] == list(range(len(records)))
        results.append(
            dict(
                case=w["name"],
                workload=w,
                status_counts=dict(Counter(r["status"] for r in records)),
                pipeline=evaluate_ranking(rank_records(records), records, measured),
                traffic_waves=evaluate_ranking(traffic_ranking(records), records, measured),
                wall_seconds=s["wall_seconds"],
                stage_cost_percent=s["carver"]["stage_cost_percent"],
                error=s["error"],
                record_errors=s["record_errors"],
            )
        )
    write(root / "evaluation.json", dict(protocol_sha256=digest(root / "freeze.json"), results=results))
    print(json.dumps(results, indent=2))


def predict(root, profile_path, destination):
    """Fresh actual-IR predictions, using only workload declarations (never latencies)."""
    p = json.loads((root / "freeze.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for w in p["workloads"]:
        reason = unsupported_reason(w, p["target"])
        if reason:
            write(destination / (w["name"] + ".json"), dict(workload=w, status="unsupported", reason=reason))
            continue
        profile = load_device_profile(profile_path, input_dtype=w["dtype"], memory_regime="streaming")
        config = CarverConfig(enabled=True, device_limits=p["device_limits"], **settings(w, profile))
        records = []
        for i, c in enumerate(configs_for(w)):
            try:
                record = analyze_prim_func(
                    factory(w)(**c), config, target=p["target"], pass_configs=PASS_CONFIGS if w["family"] == "attention" else None
                )
                record.update(index=i, config=c)
            except Exception as error:
                record = dict(index=i, config=c, error=str(error))
            records.append(record)
        write(
            destination / (w["name"] + ".json"),
            dict(
                workload=w,
                analysis_version=ANALYSIS_VERSION,
                profile_sha256=digest(profile_path),
                configs=records,
                ranking=rank_records(records),
            ),
        )
        print(w["name"], "predicted", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["profile", "freeze", "verify", "worker", "run", "evaluate", "predict"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--split", choices=list(SPLITS), default="all")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--mode", choices=["disabled", "report_only", "reject"], default="report_only")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action != "profile" and args.root is None:
        parser.error("--root is required for this action")
    if args.action in ("profile", "freeze", "predict") and args.profile is None:
        parser.error("--profile is required for this action")
    if args.group_size < 1:
        parser.error("--group-size must be positive")
    if args.action == "run" and args.case is None:
        parser.error("--case is required for run")
    if args.action == "predict" and args.destination is None:
        parser.error("--destination is required for predict")
    # Resolve user paths before changing directory; source hashes are repo-relative.
    for name in ("root", "profile", "destination"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    os.chdir(REPO_ROOT)
    if args.action == "profile":
        measure_profile(args.profile, args.refresh)
    elif args.action == "freeze":
        freeze(args.root, args.profile, args.split, mode=args.mode, group_size=args.group_size)
    elif args.action == "verify":
        verify(args.root)
    elif args.action == "worker":
        worker(args.root, args.gpu, args.shard, args.shards)
    elif args.action == "run":
        p = verify(args.root)
        run(args.root, next(w for w in p["workloads"] if w["name"] == args.case), args.gpu)
    elif args.action == "predict":
        predict(args.root, args.profile, args.destination)
    else:
        evaluate(args.root)


if __name__ == "__main__":
    main()

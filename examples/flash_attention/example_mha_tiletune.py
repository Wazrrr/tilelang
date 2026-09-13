"""Analyze real causal/noncausal attention with a reusable device profile.

The normal workflow analyzes the entire supplied grid and benchmarks only one
reference config. Exhaustive validation lives in benchmark/autotune so measured
candidate latencies never enter the ranking model.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
import tilelang
from tilelang.tiletune import analyze_prim_func, anchor_latency, profile_device, query_device_limits, rank_records
from tilelang.tiletune.config import ANALYSIS_VERSION
from tilelang.tiletune.profiling.device_profile import current_target
from examples.flash_attention.example_mha_fwd_bshd import flashattn

TARGET = {"kind": "cuda", "arch": "sm_90a"}
PASS_CONFIGS = {"tl.enable_fast_math": True, "tl.enable_cuda_resource_capture": True}
# A fixed two-warpgroup reference with a modeled physical reservation and tile
# demand. The former 128-thread reference exceeds the soft demand allowance and
# cannot anchor latency. This choice does not use candidate benchmark results.
REFERENCE_CONFIG = dict(block_M=128, block_N=128, num_stages=2, threads=256)
DEFAULT_SPILL_BUDGET_REGISTERS_PER_THREAD = 32


def get_configs():
    keys = ("block_M", "block_N", "num_stages", "threads")
    values = itertools.product([32, 64, 128, 256], [32, 64, 128, 256], [0, 1, 2, 3], [128, 256])
    return [dict(zip(keys, item)) for item in values]


def make_attention(batch=1, heads=16, sequence=4096, dim=128, causal=False):
    def attention(block_M, block_N, num_stages, threads):
        return flashattn.jit_impl.get_tir(
            batch=batch,
            heads=heads,
            seq_len=sequence,
            dim=dim,
            is_causal=causal,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
        )

    return attention


def make_inputs(batch, heads, sequence, dim, seed=123):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn((batch, sequence, heads, dim), generator=generator, device="cuda", dtype=torch.float16) for _ in range(3)]


def reference_attention(q, k, v, causal):
    """FP32 reference, chunked over queries to bound temporary memory."""
    torch.backends.cuda.matmul.allow_tf32 = False
    qh, kh, vh = [x.permute(0, 2, 1, 3).float() for x in (q, k, v)]
    output = torch.empty_like(qh)
    sequence, dim = q.shape[1], q.shape[-1]
    keys = torch.arange(sequence, device=q.device)
    for start in range(0, sequence, 256):
        end = min(sequence, start + 256)
        scores = (qh[:, :, start:end] @ kh.transpose(-1, -2)) * dim**-0.5
        if causal:
            queries = torch.arange(start, end, device=q.device)
            scores.masked_fill_(keys[None, :] > queries[:, None], -float("inf"))
        output[:, :, start:end] = scores.softmax(-1) @ vh
    return output.permute(0, 2, 1, 3).contiguous()


def check_accuracy(output, reference, rtol=0.02, atol=0.02):
    actual = output.float()
    error = (actual - reference).abs()
    mismatch = error > atol + rtol * reference.abs()
    result = dict(
        max_abs_error=error.max().item(),
        rms_error=error.square().mean().sqrt().item(),
        mismatched_elements=mismatch.sum().item(),
        finite=bool(torch.isfinite(actual).all().item()),
        rtol=rtol,
        atol=atol,
    )
    if not result["finite"] or result["mismatched_elements"]:
        raise AssertionError(f"Attention correctness failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--memory-regime", choices=["cached", "streaming"], default="streaming")
    parser.add_argument("--device-profile", type=Path, default=Path("attention_device_profile.json"))
    parser.add_argument("--output", type=Path, default=Path("attention_tiletune.json"))
    parser.add_argument("--reference-latency-ms", type=float, help="Reuse a reference timing instead of benchmarking it")
    parser.add_argument("--reference-index", type=int, help="Original config index to benchmark once for a global latency scale")
    parser.add_argument(
        "--spill-budget-registers-per-thread",
        type=int,
        default=DEFAULT_SPILL_BUDGET_REGISTERS_PER_THREAD,
        help="Soft tile-demand allowance in 32-bit registers per computing thread; zero gives a strict comparison",
    )
    args = parser.parse_args()
    target = current_target()
    start = time.perf_counter()
    profile = profile_device(input_dtype="float16", cache_path=args.device_profile, memory_regime=args.memory_regime)
    profile_seconds = time.perf_counter() - start
    factory = make_attention(args.batch, args.heads, args.sequence, args.dim, args.causal)
    limits = query_device_limits(target)
    configs = get_configs()
    reference_index = configs.index(REFERENCE_CONFIG) if args.reference_index is None else args.reference_index
    if not 0 <= reference_index < len(configs):
        parser.error(f"--reference-index must be between 0 and {len(configs) - 1}")
    settings = dict(
        mode="report_only",
        ranking_metric="pipeline_time",
        performance_model=profile,
        max_spill_bytes=None,
        max_local_bytes=None,
        attention_spill_budget_registers_per_thread=args.spill_budget_registers_per_thread,
    )
    records, reference_func = [], None
    start = time.perf_counter()
    for index, config in enumerate(configs):
        func = factory(**config)
        records.append(
            dict(
                index=index,
                config=config,
                **analyze_prim_func(func, settings, target=target, device_limits=limits, pass_configs=PASS_CONFIGS),
            )
        )
        if index == reference_index:
            reference_func = func
    report = dict(
        workload=dict(
            batch=args.batch, heads=args.heads, sequence=args.sequence, dim=args.dim, causal=args.causal, dtype="float16", layout="BSHD"
        ),
        analysis_version=ANALYSIS_VERSION,
        settings=settings,
        profile=profile,
        profile_seconds=profile_seconds,
        analysis_seconds=time.perf_counter() - start,
        source_sha256={
            p.relative_to("tilelang/tiletune").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("tilelang/tiletune").rglob("*.py")
        },
        configs=configs,
        records=records,
        ranking=rank_records(records),
        reference_index=reference_index,
        reference_config=configs[reference_index],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    latency = args.reference_latency_ms
    if latency is None:
        inputs = make_inputs(args.batch, args.heads, args.sequence, args.dim)
        reference = reference_attention(*inputs, args.causal)
        kernel = tilelang.compile(reference_func, target=target, out_idx=[3], execution_backend="tvm_ffi", pass_configs=PASS_CONFIGS)
        report["reference_accuracy"] = check_accuracy(kernel(*inputs), reference)
        latency = kernel.get_profiler().do_bench(input_tensors=inputs, n_warmup=5, n_repeat=30, backend="cudagraph")
    report["reference_latency_ms"] = latency
    report["target_configs_benchmarked"] = int(args.reference_latency_ms is None)
    if records[reference_index]["tile_cost"].get("score") is not None:
        report["anchored_profile"] = anchor_latency(profile, records[reference_index], latency)
    else:
        report["anchor_status"] = "unknown: reference has no pipeline timing score"
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(dict(causal=args.causal, reference_latency_ms=latency, top=report["ranking"][:8], output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()

"""Exhaustive attention validation with frozen generic device measurements."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from benchmark.autotune.benchmark_tiletune_gemm import ObservedTuner, snapshot
from examples.flash_attention.example_mha_tiletune import (
    DEFAULT_SPILL_BUDGET_REGISTERS_PER_THREAD,
    PASS_CONFIGS,
    check_accuracy,
    get_configs,
    make_attention,
    make_inputs,
    reference_attention,
)
from tilelang.tiletune import load_device_profile
from tilelang.tiletune.device_profile import current_target, _identity
from tilelang.tiletune.config import ANALYSIS_VERSION


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


class AttentionTuner(ObservedTuner):
    def _prepare_compile_execution(self, **kwargs):
        result = super()._prepare_compile_execution(**kwargs)
        for future in result[1]:

            def checkpoint(done):
                try:
                    for index, _, _, _ in done.result():
                        record = dict(self.tiletune_session.records[index] if self.tiletune_session else self.outcomes[index])
                        write(self.run_dir / f"compiled_{index}.json", record)
                except Exception as error:
                    self.record_errors.append(f"checkpoint: {error}")

            future.add_done_callback(checkpoint)
        return result

    def _write_benchmark_result(self, idx, status, latency, error):
        super()._write_benchmark_result(idx, status, latency, error)
        write(self.run_dir / f"result_{idx}.json", self.outcomes[idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--device-profile", type=Path, required=True)
    parser.add_argument("--memory-regime", choices=["cached", "streaming"], default="streaming")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["disabled", "report_only", "reject"], default="report_only")
    parser.add_argument("--max-spill-bytes", type=int, default=None, help="Optional spill limit; attention allows spills by default")
    parser.add_argument(
        "--max-local-bytes", type=int, default=None, help="Optional local-memory limit; attention allows local memory by default"
    )
    parser.add_argument("--indices", help="Original config indices for targeted validation")
    parser.add_argument(
        "--spill-budget-registers-per-thread",
        type=int,
        default=DEFAULT_SPILL_BUDGET_REGISTERS_PER_THREAD,
        help="Soft tile-demand allowance in 32-bit registers per computing thread; zero gives a strict comparison",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_AUTO_TUNING_CPU_COUNTS"] = str(args.workers)
    os.environ["TILELANG_AUTOTUNE_TIMING_LOG"] = str(args.output / "timings.tsv")
    profile = load_device_profile(
        args.device_profile, input_dtype="float16", expected_identity=_identity(), memory_regime=args.memory_regime
    )
    grid = get_configs()
    indices = list(map(int, args.indices.split(","))) if args.indices else list(range(len(grid)))
    configs = [grid[i] for i in indices]
    source_paths = list(Path("tilelang/tiletune").rglob("*.py")) + [
        Path(__file__),
        Path("examples/flash_attention/example_mha_tiletune.py"),
        Path("examples/flash_attention/example_mha_fwd_bshd.py"),
        Path("src/cuda/transform/producer_consumer_ws.cc"),
    ]
    write(
        args.output / "frozen.json",
        dict(
            workload=dict(
                batch=args.batch, heads=args.heads, sequence=args.sequence, dim=args.dim, causal=args.causal, dtype="float16", layout="BSHD"
            ),
            profile=profile,
            configs=configs,
            original_indices=indices,
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        ),
    )
    inputs = make_inputs(args.batch, args.heads, args.sequence, args.dim)
    reference = reference_attention(*inputs, args.causal)
    accuracies = []

    def check(actuals, refs):
        accuracies.append(check_accuracy(actuals[0], refs[0]))

    tuner = (
        AttentionTuner(make_attention(args.batch, args.heads, args.sequence, args.dim, args.causal), configs)
        .set_compile_args(target=current_target(), execution_backend="tvm_ffi", out_idx=[3], pass_configs=PASS_CONFIGS)
        .set_profile_args(
            ref_prog=lambda q, k, v: reference,
            supply_prog=lambda params: inputs,
            manual_check_prog=check,
            backend="cudagraph",
            cache_input_tensors=True,
        )
        .set_benchmark_report_path(str(args.output / "benchmarks.tsv"))
    )
    if args.mode != "disabled":
        tuner.set_tiletune_args(
            True,
            mode=args.mode,
            report_path=str(args.output / "tiletune.json"),
            ranking_metric="pipeline_time",
            performance_model=profile,
            max_spill_bytes=args.max_spill_bytes,
            max_local_bytes=args.max_local_bytes,
            attention_spill_budget_registers_per_thread=args.spill_budget_registers_per_thread,
        )
    tuner.run_dir = args.output
    tuner.record_errors = []
    tuner.outcomes = [
        dict(index=i, original_index=indices[i], config=c, compile_status="pending", benchmark_status="not_run")
        for i, c in enumerate(configs)
    ]
    before, started = snapshot(args.gpu), time.perf_counter()
    result, error = None, None
    try:
        result = tuner.run(
            warmup=args.warmup,
            rep=args.rep,
            timeout=30,
            early_stop=False,
            use_pipeline=False,
            enable_grouped_compile=args.group_size > 1,
            group_compile_size=args.group_size,
        )
    except Exception as exc:
        error = str(exc)
    summary = dict(
        causal=args.causal,
        mode=args.mode,
        gpu=args.gpu,
        analysis_version=ANALYSIS_VERSION,
        config_count=len(configs),
        group_size=args.group_size,
        wall_seconds=time.perf_counter() - started,
        gpu_before=before,
        gpu_after=snapshot(args.gpu),
        error=error,
        profile=profile,
        configs=tuner.outcomes,
        tiletune=tuner.tiletune_report,
        record_errors=tuner.record_errors,
        correctness=accuracies,
        winner_config=result.config if result else None,
        winner_latency_ms=result.latency if result else None,
    )
    write(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("causal", "config_count", "wall_seconds", "error", "winner_config", "winner_latency_ms")}))
    if error or tuner.record_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

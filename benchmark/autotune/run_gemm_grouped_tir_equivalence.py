"""Compare pre-codegen device TIR between normal and grouped compilation paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SHAPES = ["4096x4096x4096"]
GROUPED_SYMBOL_SUFFIX_RE = re.compile(r"_gc_[0-9]+")


def parse_shape(shape_text: str) -> tuple[int, int, int]:
    parts = shape_text.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid shape '{shape_text}', expected format like 4096x4096x4096")
    m, n, k = (int(part.strip()) for part in parts)
    return m, n, k


def canonicalize_prim_func(func: Any) -> str:
    from tilelang import tvm
    from tvm import tir

    normalized_func = func
    if isinstance(func, tir.PrimFunc) and func.attrs is not None and "global_symbol" in func.attrs:
        normalized_func = func.without_attr("global_symbol")
    json_text = tvm.ir.save_json(normalized_func)
    json_text = GROUPED_SYMBOL_SUFFIX_RE.sub("_gc_X", json_text)
    return json_text


def module_fingerprint(mod: Any) -> tuple[str, list[str]]:
    from tilelang import tvm

    per_func_digests: list[str] = []
    for _, func in mod.functions.items():
        canonical_text = canonicalize_prim_func(func)
        per_func_digests.append(hashlib.sha256(canonical_text.encode("utf-8")).hexdigest())
    per_func_digests.sort()
    joined = "||".join(per_func_digests)
    module_digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return module_digest, per_func_digests


def lower_program(
    program: Any,
    target: Any,
    target_host: Any,
    pass_configs: dict[str, Any],
) -> tuple[Any, Any]:
    from tilelang import tvm
    from tilelang.engine.lower import lower_to_host_device_ir

    with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
        host_mod, device_mod, _params, _norm_target, _norm_target_host, _stage = lower_to_host_device_ir(
            program,
            target=target,
            target_host=target_host,
        )
    return host_mod, device_mod


def chunk_units(configs: list[dict[str, Any]], group_size: int) -> list[list[tuple[int, dict[str, Any]]]]:
    units: list[list[tuple[int, dict[str, Any]]]] = []
    for start in range(0, len(configs), group_size):
        end = min(start + group_size, len(configs))
        units.append([(i, configs[i]) for i in range(start, end)])
    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Shape MxNxK. Repeat for multiple shapes.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/gemm_grouped_tir_equivalence.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--mismatch-dir",
        type=Path,
        default=Path("results/gemm_grouped_tir_mismatch"),
        help="Directory to dump mismatched device TIR scripts.",
    )
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="tvm_ffi",
        choices=["auto", "tvm_ffi", "cython", "nvrtc", "torch"],
        help="Compile backend used to build autotuner/kernel context.",
    )
    parser.add_argument(
        "--profile-backend",
        type=str,
        default="event",
        choices=["event", "cupti", "cudagraph"],
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=20,
        help="Max number of configs per shape (<=0 means all configs).",
    )
    parser.add_argument(
        "--group-compile-size",
        type=int,
        default=4,
        help="Number of configs in each grouped compile unit.",
    )

    roller_group = parser.add_mutually_exclusive_group()
    roller_group.add_argument("--with-roller", dest="with_roller", action="store_true")
    roller_group.add_argument("--without-roller", dest="with_roller", action="store_false")
    parser.set_defaults(with_roller=True)

    disable_cache_group = parser.add_mutually_exclusive_group()
    disable_cache_group.add_argument("--disable-cache", dest="disable_cache", action="store_true")
    disable_cache_group.add_argument("--enable-cache", dest="disable_cache", action="store_false")
    parser.set_defaults(disable_cache=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from examples.gemm.example_gemm_autotune import _build_autotuner
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.max_configs == 0:
        raise ValueError("--max-configs cannot be 0, use negative for all configs")
    if args.group_compile_size <= 0:
        raise ValueError("--group-compile-size must be > 0")

    if args.disable_cache:
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"
    else:
        os.environ["TILELANG_DISABLE_CACHE"] = "0"
        os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "0"

    shape_texts = args.shape if args.shape else DEFAULT_SHAPES
    shapes = [parse_shape(shape_text) for shape_text in shape_texts]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.mismatch_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "shape",
        "config_idx",
        "status",
        "error",
        "same_device_tir_before_codegen",
        "normal_device_func_count",
        "grouped_device_func_count",
        "normal_device_digest",
        "grouped_device_digest",
        "group_compile_size",
        "group_unit_idx",
        "group_unit_size",
        "execution_backend",
        "profile_backend",
        "with_roller",
        "topk",
        "max_configs",
        "config",
    ]

    total_rows = 0
    mismatch_count = 0

    with args.csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for shape_idx, (m, n, k) in enumerate(shapes, start=1):
            shape_text = f"{m}x{n}x{k}"
            print(f"[shape {shape_idx}/{len(shapes)}] shape={shape_text}")

            autotuner, configs, kernel = _build_autotuner(
                M=m,
                N=n,
                K=k,
                with_roller=args.with_roller,
                profile_backend=args.profile_backend,
                execution_backend=args.execution_backend,
                skip_check=True,
                cache_input_tensors=False,
                topk=args.topk,
            )

            compile_args = autotuner.compile_args
            pass_configs = dict(compile_args.pass_configs) if compile_args.pass_configs else {}
            target_kind = compile_args.target.kind.name
            if target_kind != "cuda" or str(compile_args.execution_backend) != "tvm_ffi":
                raise RuntimeError(
                    "This experiment requires CUDA + tvm_ffi because it must call compile_grouped_unit_tvm_ffi directly. "
                    f"Current target={target_kind}, execution_backend={compile_args.execution_backend}."
                )

            selected_configs = configs
            if args.max_configs > 0:
                selected_configs = configs[:args.max_configs]
            print(f"  selected_configs={len(selected_configs)} total_available={len(configs)}")

            # Normal path: pre-codegen lowering per config
            normal_by_idx: dict[int, tuple[str, list[str], str]] = {}
            for config_idx, config in enumerate(selected_configs):
                try:
                    normal_program = kernel(**config)
                    _normal_host_mod, normal_device_mod = lower_program(
                        normal_program,
                        target=compile_args.target,
                        target_host=compile_args.target_host,
                        pass_configs=pass_configs,
                    )
                    normal_device_digest, normal_device_func_digests = module_fingerprint(normal_device_mod)
                    normal_by_idx[config_idx] = (
                        normal_device_digest,
                        normal_device_func_digests,
                        normal_device_mod.script(),
                    )
                except Exception as ex:
                    row = {
                        "shape": shape_text,
                        "config_idx": config_idx,
                        "status": "failed",
                        "error": f"normal_lower_failed: {type(ex).__name__}: {ex}",
                        "same_device_tir_before_codegen": "",
                        "normal_device_func_count": "",
                        "grouped_device_func_count": "",
                        "normal_device_digest": "",
                        "grouped_device_digest": "",
                        "group_compile_size": args.group_compile_size,
                        "group_unit_idx": "",
                        "group_unit_size": "",
                        "execution_backend": args.execution_backend,
                        "profile_backend": args.profile_backend,
                        "with_roller": args.with_roller,
                        "topk": args.topk,
                        "max_configs": args.max_configs,
                        "config": json.dumps(config, sort_keys=True),
                    }
                    writer.writerow(row)
                    fp.flush()
                    total_rows += 1
                    print(f"  [cfg {config_idx}] failed: {row['error']}")

            def elaborate_func(**cfg):
                return kernel(**cfg)

            grouped_units = chunk_units(selected_configs, args.group_compile_size)
            for unit_idx, unit_items in enumerate(grouped_units):
                unit_results = compile_grouped_unit_tvm_ffi(
                    unit_items=unit_items,
                    compile_args=compile_args,
                    elaborate_func=elaborate_func,
                )

                for config_idx, config, jit_kernel, grouped_error, _compile_measurement in unit_results:
                    row = {
                        "shape": shape_text,
                        "config_idx": config_idx,
                        "status": "ok",
                        "error": "",
                        "same_device_tir_before_codegen": "",
                        "normal_device_func_count": "",
                        "grouped_device_func_count": "",
                        "normal_device_digest": "",
                        "grouped_device_digest": "",
                        "group_compile_size": args.group_compile_size,
                        "group_unit_idx": unit_idx,
                        "group_unit_size": len(unit_items),
                        "execution_backend": args.execution_backend,
                        "profile_backend": args.profile_backend,
                        "with_roller": args.with_roller,
                        "topk": args.topk,
                        "max_configs": args.max_configs,
                        "config": json.dumps(config, sort_keys=True),
                    }

                    if grouped_error is not None or jit_kernel is None:
                        row["status"] = "failed"
                        row["error"] = f"grouped_compile_failed: {grouped_error}"
                        writer.writerow(row)
                        fp.flush()
                        total_rows += 1
                        print(f"  [cfg {config_idx}] failed: {row['error']}")
                        continue

                    if config_idx not in normal_by_idx:
                        row["status"] = "failed"
                        row["error"] = "normal_baseline_missing"
                        writer.writerow(row)
                        fp.flush()
                        total_rows += 1
                        print(f"  [cfg {config_idx}] failed: {row['error']}")
                        continue

                    grouped_device_mod = jit_kernel.artifact.device_mod
                    grouped_device_digest, grouped_device_func_digests = module_fingerprint(grouped_device_mod)
                    grouped_device_script = grouped_device_mod.script()

                    normal_device_digest, normal_device_func_digests, normal_device_script = normal_by_idx[config_idx]
                    same_device = normal_device_func_digests == grouped_device_func_digests

                    row["same_device_tir_before_codegen"] = same_device
                    row["normal_device_func_count"] = len(normal_device_func_digests)
                    row["grouped_device_func_count"] = len(grouped_device_func_digests)
                    row["normal_device_digest"] = normal_device_digest
                    row["grouped_device_digest"] = grouped_device_digest

                    writer.writerow(row)
                    fp.flush()
                    total_rows += 1

                    if not same_device:
                        mismatch_count += 1
                        base = f"{shape_text.replace('x', '_')}_cfg_{config_idx:04d}"
                        normal_path = args.mismatch_dir / f"{base}.normal_device_pre_codegen.tir"
                        grouped_path = args.mismatch_dir / f"{base}.grouped_device_pre_codegen.tir"
                        normal_path.write_text(normal_device_script)
                        grouped_path.write_text(grouped_device_script)
                        print(f"  [cfg {config_idx}] device_tir_mismatch -> {normal_path.name}, {grouped_path.name}")
                    else:
                        print(f"  [cfg {config_idx}] device_pre_codegen_equal=True")

    print(f"Finished. rows={total_rows} device_tir_mismatches={mismatch_count} csv={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

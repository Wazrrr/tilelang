"""Dump and compare pre-codegen device TIR for vector_add in normal vs grouped paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GROUPED_SYMBOL_SUFFIX_RE = re.compile(r"_gc_[0-9]+")


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
    per_func_digests: list[str] = []
    for _, func in mod.functions.items():
        canonical_text = canonicalize_prim_func(func)
        per_func_digests.append(hashlib.sha256(canonical_text.encode("utf-8")).hexdigest())
    per_func_digests.sort()
    module_digest = hashlib.sha256("||".join(per_func_digests).encode("utf-8")).hexdigest()
    return module_digest, per_func_digests


def chunk_units(configs: list[dict[str, Any]], group_size: int) -> list[list[tuple[int, dict[str, Any]]]]:
    units: list[list[tuple[int, dict[str, Any]]]] = []
    for start in range(0, len(configs), group_size):
        end = min(start + group_size, len(configs))
        units.append([(i, configs[i]) for i in range(start, end)])
    return units


def build_vector_add_kernel(N: int):
    import tilelang.language as T

    def kernel(block_N=None, threads=None):
        @T.prim_func
        def main(
            A: T.Tensor((N,), "float32"),
            B: T.Tensor((N,), "float32"),
            C: T.Tensor((N,), "float32"),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
                for i in T.Parallel(block_N):
                    idx = bx * block_N + i
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return main

    return kernel


def get_default_configs() -> list[dict[str, int]]:
    return [
        {"block_N": 128, "threads": 128},
        {"block_N": 256, "threads": 128},
        {"block_N": 256, "threads": 256},
        {"block_N": 512, "threads": 256},
        {"block_N": 1024, "threads": 256},
        {"block_N": 1024, "threads": 512},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4096, help="Vector length.")
    parser.add_argument("--max-configs", type=int, default=-1, help="Max configs to test (<=0 means all).")
    parser.add_argument("--group-compile-size", type=int, default=2)
    parser.add_argument("--target", type=str, default="auto")
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="tvm_ffi",
        choices=["auto", "tvm_ffi", "cython", "nvrtc", "torch"],
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/vector_add_grouped_tir_dump.csv"),
        help="CSV output summary path.",
    )
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path("results/vector_add_tir_dump"),
        help="Root directory for DumpIR outputs and dumped device TIR scripts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from tilelang import tvm
    from tilelang.autotuner import AutoTuner
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.engine.lower import lower_to_host_device_ir, device_codegen
    from tilelang.transform import PassConfigKey

    if args.N <= 0:
        raise ValueError("--N must be > 0")
    if args.group_compile_size <= 0:
        raise ValueError("--group-compile-size must be > 0")
    if args.max_configs == 0:
        raise ValueError("--max-configs cannot be 0, use negative for all configs")

    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"

    kernel = build_vector_add_kernel(args.N)
    configs = get_default_configs()
    if args.max_configs > 0:
        configs = configs[:args.max_configs]

    autotuner = AutoTuner.from_kernel(kernel=kernel, configs=configs).set_compile_args(
        out_idx=[-1],
        target=args.target,
        execution_backend=args.execution_backend,
    )
    compile_args = autotuner.compile_args

    target_kind = compile_args.target.kind.name
    backend = str(compile_args.execution_backend)
    if target_kind != "cuda" or backend != "tvm_ffi":
        raise RuntimeError(
            "This script is for compile_grouped_unit_tvm_ffi path and requires CUDA+tvm_ffi. "
            f"Current target={target_kind}, execution_backend={backend}."
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.dump_root.mkdir(parents=True, exist_ok=True)
    normal_dump_root = args.dump_root / "normal"
    grouped_dump_root = args.dump_root / "grouped"
    mismatch_dump_root = args.dump_root / "mismatch"
    normal_dump_root.mkdir(parents=True, exist_ok=True)
    grouped_dump_root.mkdir(parents=True, exist_ok=True)
    mismatch_dump_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "config_idx",
        "status",
        "error",
        "same_device_tir_before_codegen",
        "normal_device_digest",
        "grouped_device_digest",
        "normal_device_func_count",
        "grouped_device_func_count",
        "group_unit_idx",
        "group_unit_size",
        "normal_dump_dir",
        "grouped_dump_dir",
        "normal_device_tir_file",
        "grouped_device_tir_file",
        "normal_device_kernel_source_file",
        "grouped_device_kernel_source_file",
        "config",
    ]

    # Baseline per-config lowering with dump_ir
    normal_by_idx: dict[int, dict[str, Any]] = {}
    for config_idx, config in enumerate(configs):
        normal_dump_dir = normal_dump_root / f"cfg_{config_idx:04d}"
        normal_dump_dir.mkdir(parents=True, exist_ok=True)
        pass_configs = {
            PassConfigKey.TL_ENABLE_DUMP_IR: True,
            PassConfigKey.TL_DUMP_IR_DIR: str(normal_dump_dir),
        }
        try:
            program = kernel(**config)
            with tvm.transform.PassContext(opt_level=3, config=pass_configs), compile_args.target:
                _host_mod, device_mod, _params, _norm_target, _norm_target_host, _stage = lower_to_host_device_ir(
                    program,
                    target=compile_args.target,
                    target_host=compile_args.target_host,
                )
                normal_rt_mod = device_codegen(device_mod, compile_args.target)
                normal_kernel_source = normal_rt_mod.inspect_source()
            digest, per_func_digests = module_fingerprint(device_mod)
            device_tir_file = normal_dump_dir / "device_pre_codegen.tir"
            device_tir_file.write_text(device_mod.script())
            kernel_source_file = normal_dump_dir / "device_kernel_source.cu"
            kernel_source_file.write_text(normal_kernel_source)
            normal_by_idx[config_idx] = {
                "digest": digest,
                "per_func_digests": per_func_digests,
                "device_mod": device_mod,
                "dump_dir": normal_dump_dir,
                "device_tir_file": device_tir_file,
                "kernel_source_file": kernel_source_file,
            }
            print(f"[normal cfg={config_idx}] lowered + dumped -> {normal_dump_dir}")
        except Exception as ex:
            normal_by_idx[config_idx] = {
                "error": f"{type(ex).__name__}: {ex}",
                "dump_dir": normal_dump_dir,
            }
            print(f"[normal cfg={config_idx}] failed: {normal_by_idx[config_idx]['error']}")

    rows: list[dict[str, Any]] = []
    mismatches = 0

    def elaborate_func(**cfg):
        return kernel(**cfg)

    grouped_units = chunk_units(configs, args.group_compile_size)
    for unit_idx, unit_items in enumerate(grouped_units):
        grouped_dump_dir = grouped_dump_root / f"unit_{unit_idx:04d}"
        grouped_dump_dir.mkdir(parents=True, exist_ok=True)
        grouped_kernel_source_file = grouped_dump_dir / "device_kernel_source_grouped.cu"
        pass_configs = {
            PassConfigKey.TL_ENABLE_DUMP_IR: True,
            PassConfigKey.TL_DUMP_IR_DIR: str(grouped_dump_dir),
        }
        compile_args_unit = replace(compile_args, pass_configs=pass_configs)

        unit_results = compile_grouped_unit_tvm_ffi(
            unit_items=unit_items,
            compile_args=compile_args_unit,
            elaborate_func=elaborate_func,
        )

        for config_idx, config, jit_kernel, grouped_error, _compile_measurement in unit_results:
            row = {
                "config_idx": config_idx,
                "status": "ok",
                "error": "",
                "same_device_tir_before_codegen": "",
                "normal_device_digest": "",
                "grouped_device_digest": "",
                "normal_device_func_count": "",
                "grouped_device_func_count": "",
                "group_unit_idx": unit_idx,
                "group_unit_size": len(unit_items),
                "normal_dump_dir": str(normal_by_idx.get(config_idx, {}).get("dump_dir", "")),
                "grouped_dump_dir": str(grouped_dump_dir),
                "normal_device_tir_file": str(normal_by_idx.get(config_idx, {}).get("device_tir_file", "")),
                "grouped_device_tir_file": "",
                "normal_device_kernel_source_file": str(normal_by_idx.get(config_idx, {}).get("kernel_source_file", "")),
                "grouped_device_kernel_source_file": "",
                "config": json.dumps(config, sort_keys=True),
            }

            normal_info = normal_by_idx.get(config_idx)
            if normal_info is None or "error" in normal_info:
                row["status"] = "failed"
                row["error"] = "normal_lowering_missing_or_failed"
                rows.append(row)
                print(f"[grouped cfg={config_idx}] failed: {row['error']}")
                continue

            if grouped_error is not None or jit_kernel is None:
                row["status"] = "failed"
                row["error"] = f"grouped_compile_failed: {grouped_error}"
                rows.append(row)
                print(f"[grouped cfg={config_idx}] failed: {row['error']}")
                continue

            grouped_device_mod = jit_kernel.artifact.device_mod
            grouped_digest, grouped_func_digests = module_fingerprint(grouped_device_mod)
            grouped_device_tir_file = grouped_dump_dir / f"cfg_{config_idx:04d}_device_pre_codegen.tir"
            grouped_device_tir_file.write_text(grouped_device_mod.script())
            grouped_kernel_source = jit_kernel.artifact.kernel_source
            if grouped_kernel_source is not None and not grouped_kernel_source_file.exists():
                grouped_kernel_source_file.write_text(grouped_kernel_source)

            normal_func_digests = normal_info["per_func_digests"]
            normal_digest = normal_info["digest"]
            same_device = grouped_func_digests == normal_func_digests

            row["same_device_tir_before_codegen"] = same_device
            row["normal_device_digest"] = normal_digest
            row["grouped_device_digest"] = grouped_digest
            row["normal_device_func_count"] = len(normal_func_digests)
            row["grouped_device_func_count"] = len(grouped_func_digests)
            row["grouped_device_tir_file"] = str(grouped_device_tir_file)
            if grouped_kernel_source_file.exists():
                row["grouped_device_kernel_source_file"] = str(grouped_kernel_source_file)
            rows.append(row)

            if not same_device:
                mismatches += 1
                mismatch_normal_path = mismatch_dump_root / f"cfg_{config_idx:04d}.normal_device_pre_codegen.tir"
                mismatch_grouped_path = mismatch_dump_root / f"cfg_{config_idx:04d}.grouped_device_pre_codegen.tir"
                mismatch_normal_path.write_text(normal_info["device_mod"].script())
                mismatch_grouped_path.write_text(grouped_device_mod.script())
                print(
                    f"[grouped cfg={config_idx}] mismatch -> {mismatch_normal_path.name}, {mismatch_grouped_path.name}"
                )
            else:
                print(f"[grouped cfg={config_idx}] pre_codegen_equal=True")

    with args.csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. configs={len(configs)} mismatches={mismatches} csv={args.csv}")
    print(f"Normal dump_ir dir: {normal_dump_root}")
    print(f"Grouped dump_ir dir: {grouped_dump_root}")
    print(f"Mismatch dir: {mismatch_dump_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

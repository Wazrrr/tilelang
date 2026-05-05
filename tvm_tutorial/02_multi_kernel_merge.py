#!/usr/bin/env python3
"""Tutorial 2: multiple kernels merged into one grouped device compile unit."""

from __future__ import annotations

import argparse
from pathlib import Path

from tutorial_common import (
    grouped_device_compile,
    lower_one_kernel,
    make_vec_add_kernel,
    make_vec_sub_kernel,
    normalize_target,
    parse_pass_configs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="cuda", help="Compilation target, e.g. cuda/hip/metal/c.")
    parser.add_argument("--target-host", default=None, help="Optional target_host override.")
    parser.add_argument("--n", type=int, default=1024, help="Vector length for both demo kernels.")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("tvm_tutorial/out_grouped"),
        help="Directory to write all intermediate artifacts.",
    )
    parser.add_argument(
        "--compile-device",
        action="store_true",
        help="Use full device compilation (nvcc/hipcc/metal) instead of source-only codegen.",
    )
    parser.add_argument(
        "--pass-config-json",
        type=str,
        default="{}",
        help="JSON string for TVM PassContext config, e.g. '{\"tir.disable_vectorize\": true}'.",
    )
    parser.add_argument(
        "--dispatch-trace",
        action="store_true",
        help="Emit and print dispatch trace metadata (symbol resolution/import info).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dump_dir = args.dump_dir
    dump_dir.mkdir(parents=True, exist_ok=True)

    target = normalize_target(args.target)
    pass_configs = parse_pass_configs(args.pass_config_json)

    kernel_a = make_vec_add_kernel(args.n, "tutorial_vec_add")
    kernel_b = make_vec_sub_kernel(args.n, "tutorial_vec_sub")

    lowered_a = lower_one_kernel(
        func=kernel_a,
        kernel_name="kernel_a",
        target=target,
        target_host=args.target_host,
        pass_configs=pass_configs,
        compile_device=args.compile_device,
        dump_dir=dump_dir,
    )
    lowered_b = lower_one_kernel(
        func=kernel_b,
        kernel_name="kernel_b",
        target=target,
        target_host=args.target_host,
        pass_configs=pass_configs,
        compile_device=args.compile_device,
        dump_dir=dump_dir,
    )

    grouped_summary = grouped_device_compile(
        lowered_items=[lowered_a, lowered_b],
        pass_configs=pass_configs,
        compile_device=args.compile_device,
        dump_dir=dump_dir,
        dispatch_trace=args.dispatch_trace,
    )

    summary = {
        "target": str(target),
        "target_host": args.target_host,
        "compile_device": bool(args.compile_device),
        "n": args.n,
        "pass_configs": pass_configs,
        "per_kernel": [
            lowered_a["summary"].__dict__,
            lowered_b["summary"].__dict__,
        ],
        "grouped": grouped_summary,
        "notes": [
            "This tutorial demonstrates grouped compile style for multiple kernels.",
            "Each kernel is lowered independently, then device IRModules are merged into one tvm.IRModule.",
            "Grouped device codegen is executed once; host modules are still emitted per-kernel and import the shared device module.",
        ],
    }
    write_json(dump_dir / "summary.json", summary)

    print(f"[Tutorial-2] Wrote artifacts to: {dump_dir}")
    print("[Tutorial-2] Key files:")
    print(f"  - {dump_dir / 'kernel_a.00_primfunc.tir'}")
    print(f"  - {dump_dir / 'kernel_b.00_primfunc.tir'}")
    print(f"  - {dump_dir / 'grouped.00_merged_device_ir.tir'}")
    print(f"  - {dump_dir / 'grouped.02_device_codegen.txt'}")
    print(f"  - {dump_dir / 'grouped.03_host_codegen_from_kernel_a.txt'}")
    print(f"  - {dump_dir / 'grouped.03_host_codegen_from_kernel_b.txt'}")
    print(f"  - {dump_dir / 'summary.json'}")
    if args.dispatch_trace:
        trace = grouped_summary.get("dispatch_trace", {})
        rows = trace.get("rows", [])
        print(f"  - {dump_dir / 'grouped.dispatch_trace.json'}")
        print("[Tutorial-2] Dispatch Trace:")
        for row in rows:
            name = row.get("name")
            imports = row.get("imports")
            resolved = row.get("resolved_symbols", {})
            ok_count = sum(1 for v in resolved.values() if v.get("resolved"))
            print(f"    {name}: imports={imports}, resolved={ok_count}/{len(resolved)} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

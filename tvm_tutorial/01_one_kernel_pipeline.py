#!/usr/bin/env python3
"""Tutorial 1: one kernel full pipeline (PrimFunc -> lower -> codegen)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tutorial_common import (
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
    parser.add_argument("--n", type=int, default=1024, help="Vector length.")
    parser.add_argument(
        "--kernel-op",
        default="add",
        choices=["add", "sub"],
        help="Kernel operation used in this single-kernel tutorial.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("tvm_tutorial/out_one"),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dump_dir = args.dump_dir
    dump_dir.mkdir(parents=True, exist_ok=True)

    target = normalize_target(args.target)
    pass_configs = parse_pass_configs(args.pass_config_json)

    if args.kernel_op == "add":
        kernel = make_vec_add_kernel(args.n, "tutorial_vec_add")
        kernel_name = "kernel_add"
    else:
        kernel = make_vec_sub_kernel(args.n, "tutorial_vec_sub")
        kernel_name = "kernel_sub"

    lowered = lower_one_kernel(
        func=kernel,
        kernel_name=kernel_name,
        target=target,
        target_host=args.target_host,
        pass_configs=pass_configs,
        compile_device=args.compile_device,
        dump_dir=dump_dir,
    )

    summary = {
        "target": str(target),
        "target_host": args.target_host,
        "compile_device": bool(args.compile_device),
        "n": args.n,
        "kernel_op": args.kernel_op,
        "pass_configs": pass_configs,
        "kernel": lowered["summary"].__dict__,
        "notes": [
            "This tutorial runs the complete pipeline for one kernel.",
            "Artifacts include PrimFunc, lowered host/device IR, device codegen output, and host codegen output.",
        ],
    }
    write_json(dump_dir / "summary.json", summary)

    print(f"[Tutorial-1] Wrote artifacts to: {dump_dir}")
    print("[Tutorial-1] Key files:")
    print(f"  - {dump_dir / (kernel_name + '.00_primfunc.tir')}")
    print(f"  - {dump_dir / (kernel_name + '.01_host_lowered.tir')}")
    print(f"  - {dump_dir / (kernel_name + '.02_device_lowered.tir')}")
    print(f"  - {dump_dir / (kernel_name + '.04_device_codegen.txt')}")
    print(f"  - {dump_dir / (kernel_name + '.05_host_codegen.txt')}")
    print(f"  - {dump_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

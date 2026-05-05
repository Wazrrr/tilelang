from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tilelang.language as T
from tilelang import tvm as tvm
from tvm.target import Target

from tilelang.engine.lower import (
    device_codegen,
    device_codegen_without_compile,
    host_codegen,
    lower_to_host_device_ir,
)
from tilelang.transform import PassConfigKey
from tilelang.utils.target import determine_target


@dataclass
class KernelSummary:
    name: str
    primfunc_symbol: str
    host_symbols: list[str]
    device_symbols: list[str]
    lower_stage_s: dict[str, float]
    device_source_len: int
    host_source_len: int


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def parse_pass_configs(text: str) -> dict[str, Any]:
    cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError("--pass-config-json must decode to an object/dict")
    return cfg


def normalize_target(target_text: str) -> Target:
    return Target(determine_target(target_text))


def safe_script(mod: Any) -> str:
    try:
        return mod.script()
    except Exception as err:  # pragma: no cover - diagnostic path
        return f"<script() failed: {type(err).__name__}: {err}>\n{repr(mod)}"


def safe_inspect_source(mod: Any) -> str:
    try:
        return mod.inspect_source()
    except Exception as err:  # pragma: no cover - diagnostic path
        return f"<inspect_source() failed: {type(err).__name__}: {err}>\n{repr(mod)}"


def module_symbols(mod: tvm.IRModule) -> list[str]:
    names: list[str] = []
    for gvar in mod.functions.keys():
        if hasattr(gvar, "name_hint"):
            names.append(gvar.name_hint)
        else:
            names.append(str(gvar))
    return sorted(names)


def make_vec_add_kernel(n: int, symbol: str) -> tvm.tir.PrimFunc:
    @T.prim_func
    def vec_add(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as pid:
            out[pid] = x[pid] + y[pid]

    return vec_add.with_attr("global_symbol", symbol)


def make_vec_sub_kernel(n: int, symbol: str) -> tvm.tir.PrimFunc:
    @T.prim_func
    def vec_sub(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as pid:
            out[pid] = x[pid] - y[pid]

    return vec_sub.with_attr("global_symbol", symbol)


def lower_one_kernel(
    *,
    func: tvm.tir.PrimFunc,
    kernel_name: str,
    target: Target,
    target_host: str | Target | None,
    pass_configs: dict[str, Any],
    compile_device: bool,
    dump_dir: Path,
) -> dict[str, Any]:
    pass_instruments: list[Any] = []
    if pass_configs.get(PassConfigKey.TL_ENABLE_DUMP_IR):
        ir_dump_dir = pass_configs.get(PassConfigKey.TL_DUMP_IR_DIR, str(dump_dir / "pass_dump"))
        pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=ir_dump_dir))

    primfunc_symbol = str(func.attrs["global_symbol"])
    write_text(dump_dir / f"{kernel_name}.00_primfunc.tir", func.script())

    with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), target:
        host_mod, device_mod, params, normalized_target, normalized_target_host, lower_stage = lower_to_host_device_ir(
            func,
            target=target,
            target_host=target_host,
        )

    write_text(dump_dir / f"{kernel_name}.01_host_lowered.tir", safe_script(host_mod))
    write_text(dump_dir / f"{kernel_name}.02_device_lowered.tir", safe_script(device_mod))
    write_json(dump_dir / f"{kernel_name}.03_lower_stage_seconds.json", lower_stage)

    with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), normalized_target:
        device_rt_mod = (
            device_codegen(device_mod, normalized_target)
            if compile_device
            else device_codegen_without_compile(device_mod, normalized_target)
        )

    device_source = safe_inspect_source(device_rt_mod)
    write_text(dump_dir / f"{kernel_name}.04_device_codegen.txt", device_source)

    with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), normalized_target:
        host_rt_mod = host_codegen(host_mod, normalized_target_host, target=normalized_target)
    host_rt_mod.import_module(device_rt_mod)

    host_source = safe_inspect_source(host_rt_mod)
    write_text(dump_dir / f"{kernel_name}.05_host_codegen.txt", host_source)

    summary = KernelSummary(
        name=kernel_name,
        primfunc_symbol=primfunc_symbol,
        host_symbols=module_symbols(host_mod),
        device_symbols=module_symbols(device_mod),
        lower_stage_s=lower_stage,
        device_source_len=len(device_source),
        host_source_len=len(host_source),
    )

    return {
        "func": func,
        "params": params,
        "host_mod": host_mod,
        "device_mod": device_mod,
        "target": normalized_target,
        "target_host": normalized_target_host,
        "device_rt_mod": device_rt_mod,
        "host_rt_mod": host_rt_mod,
        "summary": summary,
    }


def merge_device_modules(device_mods: list[tvm.IRModule]) -> tuple[tvm.IRModule, list[str]]:
    merged_funcs: dict[Any, Any] = {}
    merged_attrs: Any = None
    seen_symbols: set[str] = set()

    for mod in device_mods:
        if merged_attrs is None:
            merged_attrs = mod.attrs
        for gvar, tir_func in mod.functions.items():
            symbol = gvar.name_hint if hasattr(gvar, "name_hint") else str(gvar)
            if symbol in seen_symbols:
                raise RuntimeError(f"Duplicate symbol while merging device modules: {symbol}")
            seen_symbols.add(symbol)
            merged_funcs[gvar] = tir_func

    merged = tvm.IRModule(merged_funcs, attrs=merged_attrs)
    return merged, sorted(seen_symbols)


def grouped_device_compile(
    *,
    lowered_items: list[dict[str, Any]],
    pass_configs: dict[str, Any],
    compile_device: bool,
    dump_dir: Path,
    dispatch_trace: bool = False,
) -> dict[str, Any]:
    assert len(lowered_items) > 0, "lowered_items must be non-empty"

    device_mods = [item["device_mod"] for item in lowered_items]
    merged_device_mod, merged_symbols = merge_device_modules(device_mods)

    write_text(dump_dir / "grouped.00_merged_device_ir.tir", safe_script(merged_device_mod))
    write_json(dump_dir / "grouped.01_merged_device_symbols.json", merged_symbols)

    reference_target = lowered_items[0]["target"]
    reference_target_host = lowered_items[0]["target_host"]

    with tvm.transform.PassContext(opt_level=3, config=pass_configs), reference_target:
        grouped_device_rt_mod = (
            device_codegen(merged_device_mod, reference_target)
            if compile_device
            else device_codegen_without_compile(merged_device_mod, reference_target)
        )

    grouped_device_source = safe_inspect_source(grouped_device_rt_mod)
    write_text(dump_dir / "grouped.02_device_codegen.txt", grouped_device_source)

    grouped_host_summaries = []
    dispatch_trace_rows: list[dict[str, Any]] = []

    def _resolve_symbol(mod: Any, symbol: str) -> tuple[bool, str]:
        try:
            func = mod.get_function(symbol, True)
            return (func is not None), ""
        except TypeError:
            # Some runtime Module variants only accept one argument.
            try:
                func = mod.get_function(symbol)
                return (func is not None), ""
            except Exception as err:  # pragma: no cover - diagnostic path
                return False, f"{type(err).__name__}: {err}"
        except Exception as err:  # pragma: no cover - diagnostic path
            return False, f"{type(err).__name__}: {err}"

    for item in lowered_items:
        name = item["summary"].name
        with tvm.transform.PassContext(opt_level=3, config=pass_configs), reference_target:
            grouped_host_rt_mod = host_codegen(item["host_mod"], reference_target_host, target=reference_target)
        grouped_host_rt_mod.import_module(grouped_device_rt_mod)
        grouped_host_source = safe_inspect_source(grouped_host_rt_mod)
        write_text(dump_dir / f"grouped.03_host_codegen_from_{name}.txt", grouped_host_source)
        grouped_host_summaries.append(
            {
                "name": name,
                "imports": len(grouped_host_rt_mod.imports),
                "host_source_len": len(grouped_host_source),
            }
        )
        if dispatch_trace:
            symbol_resolution: dict[str, dict[str, Any]] = {}
            for symbol in merged_symbols:
                ok, err = _resolve_symbol(grouped_host_rt_mod, symbol)
                symbol_resolution[symbol] = {
                    "resolved": bool(ok),
                    "error": err,
                }
            dispatch_trace_rows.append(
                {
                    "name": name,
                    "host_entry_symbol": item["summary"].primfunc_symbol,
                    "imports": len(grouped_host_rt_mod.imports),
                    "resolved_symbols": symbol_resolution,
                }
            )

    if dispatch_trace:
        write_json(
            dump_dir / "grouped.dispatch_trace.json",
            {
                "merged_device_symbols": merged_symbols,
                "rows": dispatch_trace_rows,
            },
        )

    result: dict[str, Any] = {
        "merged_device_symbol_count": len(merged_symbols),
        "merged_device_symbols": merged_symbols,
        "grouped_device_source_len": len(grouped_device_source),
        "grouped_host_summaries": grouped_host_summaries,
    }
    if dispatch_trace:
        result["dispatch_trace"] = {
            "rows": dispatch_trace_rows,
            "trace_file": str(dump_dir / "grouped.dispatch_trace.json"),
        }
    return result

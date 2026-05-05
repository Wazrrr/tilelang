"""Grouped compilation helpers for autotuner.

This module isolates backend-aware grouped compilation logic from AutoTuner.run
so tuner.py can stay focused on orchestration.
"""

from __future__ import annotations

from typing import Any, Callable
import time

from tilelang import tvm
from tvm.tir import PrimFunc

from tilelang.autotuner.param import CompileArgs
from tilelang.engine.lower import lower_to_host_device_ir, device_codegen, host_codegen
from tilelang.engine.param import CompiledArtifact
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.transform import PassConfigKey

CompileUnitResult = tuple[int, dict[str, Any], JITKernel | None, Exception | None, dict[str, float]]


def compile_grouped_unit_tvm_ffi(
    unit_items: list[tuple[int, dict[str, Any]]],
    compile_args: CompileArgs,
    elaborate_func: Callable[..., PrimFunc],
) -> list[CompileUnitResult]:
    """Compile one grouped unit for CUDA+tvm_ffi backend.

    Flow:
    1. Elaborate each config into a PrimFunc.
    2. Lower each PrimFunc into host/device IR modules.
    3. Merge all device IR into one IRModule and compile device code once.
    4. Build host runtime module per config and import shared device module.
    5. Construct per-config JITKernel objects and attach compile measurements.
    """

    pass_configs = dict(compile_args.pass_configs) if compile_args.pass_configs else {}
    pass_instruments = []
    if pass_configs.get(PassConfigKey.TL_ENABLE_DUMP_IR):
        dump_ir_path = pass_configs.get(PassConfigKey.TL_DUMP_IR_DIR, "./dump_ir")
        pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=dump_ir_path))

    unit_results: list[CompileUnitResult] = []
    lowered_items: list[dict[str, Any]] = []

    for idx, config_arg in unit_items:
        config_compile_start = time.perf_counter()
        compile_stage_measurement: dict[str, float] = {}
        try:
            build_program_start = time.perf_counter()
            program = elaborate_func(**config_arg)
            if not isinstance(program, PrimFunc):
                raise TypeError(f"Expected PrimFunc from elaboration, but got {type(program)}")
            compile_stage_measurement["compile.build_program_s"] = time.perf_counter() - build_program_start

            original_symbol = str(program.attrs["global_symbol"])
            unique_symbol = f"{original_symbol}_gc_{idx}"
            program = program.with_attr("global_symbol", unique_symbol)

            lower_start = time.perf_counter()
            with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), compile_args.target:
                host_mod, device_mod, params, normalized_target, normalized_target_host, lower_stage = lower_to_host_device_ir(
                    program,
                    target=compile_args.target,
                    target_host=compile_args.target_host,
                )
            compile_stage_measurement["compile.lower_to_host_device_s"] = time.perf_counter() - lower_start
            for stage_name, stage_value in lower_stage.items():
                if isinstance(stage_value, (int, float)):
                    compile_stage_measurement[f"lower.{stage_name}"] = float(stage_value)

            lowered_items.append(
                {
                    "idx": idx,
                    "config_arg": config_arg,
                    "program": program,
                    "host_mod": host_mod,
                    "device_mod": device_mod,
                    "params": params,
                    "target": normalized_target,
                    "target_host": normalized_target_host,
                    "compile_stage_measurement": compile_stage_measurement,
                    "config_compile_start": config_compile_start,
                }
            )
        except Exception as e:
            compile_stage_measurement["compile_unit.config_total_s"] = time.perf_counter() - config_compile_start
            unit_results.append((idx, config_arg, None, e, compile_stage_measurement))

    if not lowered_items:
        return unit_results

    try:
        merged_device_start = time.perf_counter()
        merged_funcs: dict[Any, Any] = {}
        merged_attrs = None
        merged_names: set[str] = set()
        for item in lowered_items:
            device_mod = item["device_mod"]
            if merged_attrs is None:
                merged_attrs = device_mod.attrs
            for global_var, func in device_mod.functions.items():
                name_hint = getattr(global_var, "name_hint", str(global_var))
                if name_hint in merged_names:
                    raise RuntimeError(
                        f"Duplicate device global symbol '{name_hint}' during grouped compilation "
                        f"(config index={item['idx']})."
                    )
                merged_names.add(name_hint)
                merged_funcs[global_var] = func
        merged_device_mod = tvm.IRModule(merged_funcs, attrs=merged_attrs)

        reference_target = lowered_items[0]["target"]
        with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), reference_target:
            grouped_device_rt_mod = device_codegen(merged_device_mod, reference_target)
        grouped_device_codegen_s = time.perf_counter() - merged_device_start

        grouped_kernel_source = grouped_device_rt_mod.inspect_source()
        grouped_device_codegen_share_s = grouped_device_codegen_s / len(lowered_items)

        for item in lowered_items:
            idx = item["idx"]
            config_arg = item["config_arg"]
            config_compile_start = item["config_compile_start"]
            compile_stage_measurement = item["compile_stage_measurement"]
            try:
                host_codegen_start = time.perf_counter()
                with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), item["target"]:
                    grouped_host_rt_mod = host_codegen(item["host_mod"], item["target_host"], target=item["target"])
                compile_stage_measurement["compile.group.host_codegen_s"] = time.perf_counter() - host_codegen_start

                import_module_start = time.perf_counter()
                grouped_host_rt_mod.import_module(grouped_device_rt_mod)
                compile_stage_measurement["compile.group.host_import_device_module_s"] = time.perf_counter() - import_module_start

                artifact = CompiledArtifact(
                    host_mod=grouped_host_rt_mod,
                    device_mod=item["device_mod"],
                    params=item["params"],
                    kernel_source=grouped_kernel_source,
                    rt_mod=grouped_host_rt_mod,
                )

                adapter_create_start = time.perf_counter()
                adapter = TVMFFIKernelAdapter(
                    params=artifact.params,
                    result_idx=compile_args.out_idx,
                    target=compile_args.target,
                    func_or_mod=item["program"],
                    host_mod=artifact.host_mod,
                    device_mod=artifact.device_mod,
                    rt_mod=artifact.rt_mod,
                    device_kernel_source=artifact.kernel_source,
                    verbose=compile_args.verbose,
                    pass_configs=pass_configs,
                )
                compile_stage_measurement["compile.group.adapter_create_s"] = time.perf_counter() - adapter_create_start
                compile_stage_measurement["compile.group.device_codegen_share_s"] = grouped_device_codegen_share_s

                jit_kernel = JITKernel(
                    func=item["program"],
                    out_idx=compile_args.out_idx,
                    execution_backend=compile_args.execution_backend,
                    target=compile_args.target,
                    target_host=compile_args.target_host,
                    verbose=compile_args.verbose,
                    pass_configs=pass_configs,
                    from_database=True,
                )
                jit_kernel.artifact = artifact
                jit_kernel.adapter = adapter
                jit_kernel.torch_function = adapter.func
                compile_stage_measurement["compile_unit.config_total_s"] = time.perf_counter() - config_compile_start
                jit_kernel.compile_measurement = compile_stage_measurement

                unit_results.append((idx, config_arg, jit_kernel, None, compile_stage_measurement))
            except Exception as e:
                compile_stage_measurement["compile_unit.config_total_s"] = time.perf_counter() - config_compile_start
                unit_results.append((idx, config_arg, None, e, compile_stage_measurement))
    except Exception as e:
        for item in lowered_items:
            compile_stage_measurement = item["compile_stage_measurement"]
            compile_stage_measurement["compile_unit.config_total_s"] = time.perf_counter() - item["config_compile_start"]
            unit_results.append((item["idx"], item["config_arg"], None, e, compile_stage_measurement))

    return unit_results

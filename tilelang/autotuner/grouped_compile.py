"""Grouped compilation helpers for autotuner.

This module isolates backend-aware grouped compilation logic from AutoTuner.run
so tuner.py can stay focused on orchestration.
"""

from __future__ import annotations

import contextlib
from typing import Any
from collections.abc import Callable

from tilelang import tvm
from tvm.tirx import PrimFunc

from tilelang import env
from tilelang.env import resolve_pass_profile_threshold_ms
from tilelang.autotuner.param import CompileArgs
from tilelang.engine.lower import lower_to_host_device_ir, device_codegen, host_codegen
from tilelang.engine.param import CompiledArtifact
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.contrib.cuda_resource_info import pop_recorded as cuda_pop_recorded
from tilelang.contrib.cuda_resource_info import reset_recorder as cuda_reset_recorder
from tilelang.contrib import cuda_resource_info
from tilelang.autotuner.resource_filter import (
    AutotuneResourceFilterConfig,
    AutotuneResourceFilterReject,
    evaluate_post_compile_resource_filter,
    evaluate_pre_compile_resource_filter,
    extract_launch_resource_info,
    query_cuda_device_limits,
)
from tilelang.autotuner.quality_filter import (
    AutotuneQualityFilterConfig,
    AutotuneQualityFilterReject,
    evaluate_post_compile_quality_filter,
)
from tilelang.transform import PassConfigKey
from tilelang.utils.pass_timing import build_pass_instruments, report_pass_timing_on_exit

CompileUnitResult = tuple[int, dict[str, Any], JITKernel | None, Exception | None]


def compile_grouped_unit_tvm_ffi(
    unit_items: list[tuple[int, dict[str, Any]]],
    compile_args: CompileArgs,
    elaborate_func: Callable[..., PrimFunc],
    resource_filter_config: AutotuneResourceFilterConfig | None = None,
    quality_filter_config: AutotuneQualityFilterConfig | None = None,
) -> list[CompileUnitResult]:
    """Compile one grouped unit for CUDA+tvm_ffi backend.

    Flow:
    1. Elaborate each config into a PrimFunc.
    2. Lower each PrimFunc into host/device IR modules.
    3. Merge all device IR into one IRModule and compile device code once.
    4. Build host runtime module per config and import shared device module.
    5. Construct per-config JITKernel objects that share the grouped device module.
    """

    resource_filter_config = AutotuneResourceFilterConfig.from_value(resource_filter_config)
    quality_filter_config = AutotuneQualityFilterConfig.from_value(quality_filter_config)
    pass_configs = dict(compile_args.pass_configs) if compile_args.pass_configs else {}
    device_limits = query_cuda_device_limits(resource_filter_config.device_id) if resource_filter_config.enabled else None
    base_pass_instruments = []
    if pass_configs.get(PassConfigKey.TL_ENABLE_DUMP_IR):
        dump_ir_path = pass_configs.get(PassConfigKey.TL_DUMP_IR_DIR, "./dump_ir")
        base_pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=dump_ir_path))

    enable_profile = pass_configs.get(PassConfigKey.TL_PASS_PROFILE) or env.is_pass_profile_enabled()
    profile_threshold_ms = None
    if enable_profile:
        profile_threshold_ms = resolve_pass_profile_threshold_ms(
            pass_configs,
            PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS,
            env.get_pass_profile_threshold_ms,
        )

    def create_pass_instruments():
        return build_pass_instruments(base_pass_instruments, profile_threshold_ms)

    unit_results: list[CompileUnitResult] = []
    lowered_items: list[dict[str, Any]] = []

    for idx, config_arg in unit_items:
        try:
            program = elaborate_func(**config_arg)
            original_symbol = str(program.attrs["global_symbol"])
            unique_symbol = f"{original_symbol}_gc_{idx}"
            program = program.with_attr("global_symbol", unique_symbol)

            config_instruments, timing_inst = create_pass_instruments()

            with (
                report_pass_timing_on_exit(
                    timing_inst,
                    context=f"stage=grouped-lower, config={idx}, kernel={unique_symbol}",
                ),
                tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=config_instruments),
                compile_args.target,
            ):
                host_mod, device_mod, params, normalized_target, normalized_target_host = lower_to_host_device_ir(
                    program,
                    target=compile_args.target,
                    target_host=compile_args.target_host,
                )

            launch_infos = extract_launch_resource_info(device_mod)
            filter_decisions = []
            if resource_filter_config.enabled and resource_filter_config.pre_compile:
                decision = evaluate_pre_compile_resource_filter(launch_infos, device_limits)
                filter_decisions.append(decision)
                if not decision.keep:
                    unit_results.append((idx, config_arg, None, AutotuneResourceFilterReject(decision)))
                    continue

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
                    "launch_infos": launch_infos,
                    "filter_decisions": filter_decisions,
                    "quality_decisions": [],
                }
            )
        except Exception as e:
            unit_results.append((idx, config_arg, None, e))

    if not lowered_items:
        return unit_results

    try:
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
                        f"Duplicate device global symbol '{name_hint}' during grouped compilation (config index={item['idx']})."
                    )
                merged_names.add(name_hint)
                merged_funcs[global_var] = func
        merged_device_mod = tvm.IRModule(merged_funcs, attrs=merged_attrs)

        reference_target = lowered_items[0]["target"]
        device_instruments, device_timing_inst = create_pass_instruments()
        grouped_config_indices = ",".join(str(item["idx"]) for item in lowered_items)
        capture_cuda_resources = (
            resource_filter_config.enabled
            and resource_filter_config.post_compile
        ) or quality_filter_config.needs_cuda_resource_usage()
        if capture_cuda_resources:
            cuda_reset_recorder()
        capture_context = cuda_resource_info.capture_resource_usage() if capture_cuda_resources else contextlib.nullcontext()
        try:
            with (
                capture_context,
                report_pass_timing_on_exit(
                    device_timing_inst,
                    context=f"stage=grouped-device, configs=[{grouped_config_indices}]",
                ),
                tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=device_instruments),
                reference_target,
            ):
                grouped_device_rt_mod = device_codegen(merged_device_mod, reference_target)
        finally:
            grouped_resource_usage = cuda_pop_recorded() if capture_cuda_resources else {}

        grouped_kernel_source = grouped_device_rt_mod.inspect_source()

        for item in lowered_items:
            idx = item["idx"]
            config_arg = item["config_arg"]
            try:
                if resource_filter_config.enabled and resource_filter_config.post_compile:
                    decision = evaluate_post_compile_resource_filter(item["launch_infos"], grouped_resource_usage, device_limits)
                    item["filter_decisions"].append(decision)
                    if not decision.keep:
                        unit_results.append((idx, config_arg, None, AutotuneResourceFilterReject(decision)))
                        continue

                if quality_filter_config.enabled:
                    decision = evaluate_post_compile_quality_filter(
                        launch_infos=item["launch_infos"],
                        resource_usage=grouped_resource_usage,
                        kernel_source=grouped_kernel_source,
                        config=config_arg,
                        quality_config=quality_filter_config,
                    )
                    item["quality_decisions"].append(decision)
                    if not decision.keep:
                        unit_results.append((idx, config_arg, None, AutotuneQualityFilterReject(decision)))
                        continue

                host_instruments, host_timing_inst = create_pass_instruments()
                kernel_symbol = str(item["program"].attrs["global_symbol"])
                with (
                    report_pass_timing_on_exit(
                        host_timing_inst,
                        context=f"stage=grouped-host, config={idx}, kernel={kernel_symbol}",
                    ),
                    tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=host_instruments),
                    item["target"],
                ):
                    grouped_host_rt_mod = host_codegen(item["host_mod"], item["target_host"], target=item["target"])

                grouped_host_rt_mod.import_module(grouped_device_rt_mod)

                artifact = CompiledArtifact(
                    host_mod=grouped_host_rt_mod,
                    device_mod=item["device_mod"],
                    params=item["params"],
                    kernel_source=grouped_kernel_source,
                    rt_mod=grouped_host_rt_mod,
                )

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
                if grouped_resource_usage:
                    jit_kernel._resource_usage = grouped_resource_usage
                if item["filter_decisions"]:
                    jit_kernel._resource_filter_decisions = item["filter_decisions"]
                if item["quality_decisions"]:
                    jit_kernel._quality_filter_decisions = item["quality_decisions"]

                unit_results.append((idx, config_arg, jit_kernel, None))
            except Exception as e:
                unit_results.append((idx, config_arg, None, e))
    except Exception as e:
        for item in lowered_items:
            unit_results.append((item["idx"], item["config_arg"], None, e))

    return unit_results

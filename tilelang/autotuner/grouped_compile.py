"""Grouped compilation helpers for autotuner.

This module isolates backend-aware grouped compilation logic from AutoTuner.run
so tuner.py can stay focused on orchestration.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import replace
import json
from typing import Any
from collections.abc import Callable

from tilelang import tvm
from tvm.tirx import PrimFunc

from tilelang import env
from tilelang.env import resolve_pass_profile_threshold_ms
from tilelang.autotuner.param import CompileArgs
from tilelang.engine.lower import lower_to_host_device_ir, device_codegen, device_codegen_without_compile, host_codegen
from tilelang.engine.param import CompiledArtifact
from tilelang.jit.adapter import TVMFFIKernelAdapter
from tilelang.jit.kernel import JITKernel
from tilelang.contrib.cuda_resource_info import pop_recorded as cuda_pop_recorded
from tilelang.contrib.cuda_resource_info import reset_recorder as cuda_reset_recorder
from tilelang.contrib import cuda_resource_info
from tilelang.autotuner.filters import (
    AutotuneFilterConfig,
    AutotuneFilterReject,
    evaluate_post_compile_filter,
    evaluate_pre_compile_filter,
    extract_launch_resource_info,
)
from tilelang.transform import PassConfigKey
from tilelang.utils.autotune_timing import timed_autotune_stage
from tilelang.utils.pass_timing import build_pass_instruments, report_pass_timing_on_exit

CompileUnitResult = tuple[int, dict[str, Any], JITKernel | None, Exception | None]


def compile_grouped_unit_tvm_ffi(
    unit_items: list[tuple[int, dict[str, Any]]],
    compile_args: CompileArgs,
    elaborate_func: Callable[..., PrimFunc],
    filter_config: AutotuneFilterConfig | None = None,
    carver_session=None,
    _prepared_programs=None,
) -> list[CompileUnitResult]:
    """Compile one grouped unit for CUDA+tvm_ffi backend.

    Flow:
    1. Elaborate each config into a PrimFunc.
    2. Lower each PrimFunc into host/device IR modules.
    3. Merge all device IR into one IRModule and compile device code once.
    4. Merge kept host IR, build one host runtime module, and import the shared device module.
    5. Construct per-config JITKernel objects that dispatch to named entries in the shared executable.
    """

    if carver_session is not None and _prepared_programs is None:
        # Function attributes are available only after elaboration. Split groups
        # by effective settings before lowering, keeping the same PrimFunc.
        buckets = {}
        results = []
        for idx, config_arg in unit_items:
            try:
                program = carver_session.elaborate(
                    idx, config_arg, elaborate_func, target=compile_args.target, pass_configs=compile_args.pass_configs
                )
                attrs = program.attrs or {}
                effective_pc = dict(attrs.get("tilelang_pass_configs", {}))
                effective_pc.update(compile_args.pass_configs or {})
                flags = list(attrs.get("tilelang_compile_flags", [])) + carver_session.compile_flags
                if flags:
                    key = PassConfigKey.TL_DEVICE_COMPILE_FLAGS
                    effective_pc[key] = list(effective_pc.get(key, [])) + flags
                out_idx = compile_args.out_idx
                if "tilelang_out_idx" in attrs:
                    if out_idx is not None:
                        raise ValueError("Out index conflict: out_idx and PrimFunc tilelang_out_idx are both specified")
                    out_idx = list(attrs["tilelang_out_idx"])
                effective = replace(compile_args, pass_configs=effective_pc, out_idx=out_idx)
                key = json.dumps([effective_pc, out_idx], sort_keys=True, default=str)
                bucket = buckets.setdefault(key, (effective, [], {}))
                bucket[1].append((idx, config_arg))
                bucket[2][idx] = program
                carver_session.records[idx]["effective_pass_configs"] = effective_pc
            except Exception as error:
                results.append((idx, config_arg, None, error))
        for effective, items, programs in buckets.values():
            results.extend(
                compile_grouped_unit_tvm_ffi(items, effective, elaborate_func, carver_session=carver_session, _prepared_programs=programs)
            )
        return results

    filter_config = AutotuneFilterConfig.from_value(filter_config)
    pass_configs = dict(compile_args.pass_configs) if compile_args.pass_configs else {}
    # Resource capture is a Python compilation option, not a registered TVM
    # pass option. Match JITKernel's handling before constructing PassContext.
    requested_resource_capture = bool(pass_configs.pop(cuda_resource_info.CUDA_RESOURCE_CAPTURE_CONFIG_KEY, False))
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
    unit_group_size = len(unit_items)
    unit_config_indices = ",".join(str(idx) for idx, _ in unit_items)

    for idx, config_arg in unit_items:
        try:
            with timed_autotune_stage(
                "grouped.elaborate",
                group_size=unit_group_size,
                config_idx=idx,
                configs=unit_config_indices,
            ):
                program = _prepared_programs[idx] if _prepared_programs is not None else elaborate_func(**config_arg)
                original_symbol = str(program.attrs["global_symbol"])
                unique_symbol = f"{original_symbol}_gc_{idx}"
                program = program.with_attr("global_symbol", unique_symbol)

            config_instruments, timing_inst = create_pass_instruments()

            with (
                carver_session.stage(idx, "lower") if carver_session is not None else contextlib.nullcontext(),
                timed_autotune_stage(
                    "grouped.lower",
                    group_size=unit_group_size,
                    config_idx=idx,
                    configs=unit_config_indices,
                ),
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

            with timed_autotune_stage(
                "grouped.extract_launch_info",
                group_size=unit_group_size,
                config_idx=idx,
                configs=unit_config_indices,
            ):
                launch_infos = extract_launch_resource_info(device_mod)
            filter_decisions = []
            if filter_config.enabled:
                source_instruments, source_timing_inst = create_pass_instruments()
                with (
                    timed_autotune_stage(
                        "grouped.pre_compile_codegen",
                        group_size=unit_group_size,
                        config_idx=idx,
                        configs=unit_config_indices,
                    ),
                    report_pass_timing_on_exit(
                        source_timing_inst,
                        context=f"stage=grouped-pre-compile-codegen, config={idx}, kernel={unique_symbol}",
                    ),
                    tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=source_instruments),
                    normalized_target,
                ):
                    source_mod = device_codegen_without_compile(device_mod, normalized_target)
                kernel_source = source_mod.inspect_source()

                with timed_autotune_stage(
                    "grouped.pre_compile_filter",
                    group_size=unit_group_size,
                    config_idx=idx,
                    configs=unit_config_indices,
                ):
                    decision = evaluate_pre_compile_filter(
                        launch_infos=launch_infos,
                        device_mod=device_mod,
                        config=config_arg,
                        filter_config=filter_config,
                        kernel_source=kernel_source,
                    )
                filter_decisions.append(decision)
                if not decision.keep:
                    unit_results.append(
                        (
                            idx,
                            config_arg,
                            None,
                            AutotuneFilterReject(
                                decision,
                                filter_decisions=filter_decisions,
                            ),
                        )
                    )
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
                }
            )
        except Exception as e:
            unit_results.append((idx, config_arg, None, e))

    if not lowered_items:
        return unit_results

    try:
        grouped_config_indices = ",".join(str(item["idx"]) for item in lowered_items)
        lowered_group_size = len(lowered_items)
        with timed_autotune_stage(
            "grouped.merge_device_ir",
            group_size=lowered_group_size,
            configs=grouped_config_indices,
        ):
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
        capture_cuda_resources = requested_resource_capture or filter_config.needs_cuda_resource_usage() or carver_session is not None
        if capture_cuda_resources:
            cuda_reset_recorder()
        capture_context = cuda_resource_info.capture_resource_usage() if capture_cuda_resources else contextlib.nullcontext()
        device_start = time.perf_counter()
        try:
            with (
                timed_autotune_stage(
                    "grouped.device_codegen",
                    group_size=lowered_group_size,
                    configs=grouped_config_indices,
                ),
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
            if carver_session is not None:
                duration = (time.perf_counter() - device_start) * 1000 / len(lowered_items)
                for item in lowered_items:
                    carver_session.records[item["idx"]]["timings_ms"]["device_compile"] = duration

        with timed_autotune_stage(
            "grouped.inspect_source",
            group_size=lowered_group_size,
            configs=grouped_config_indices,
        ):
            grouped_kernel_source = grouped_device_rt_mod.inspect_source()

        runtime_items: list[dict[str, Any]] = []
        for item in lowered_items:
            idx = item["idx"]
            config_arg = item["config_arg"]
            try:
                if carver_session is not None:
                    carver_session.post_compile(idx, grouped_resource_usage, item["launch_infos"], target=compile_args.target)
                if filter_config.enabled:
                    with timed_autotune_stage(
                        "grouped.post_compile_filter",
                        group_size=lowered_group_size,
                        config_idx=idx,
                        configs=grouped_config_indices,
                    ):
                        decision = evaluate_post_compile_filter(
                            launch_infos=item["launch_infos"],
                            resource_usage=grouped_resource_usage,
                            kernel_source=grouped_kernel_source,
                            config=config_arg,
                            filter_config=filter_config,
                        )
                    item["filter_decisions"].append(decision)
                    if not decision.keep:
                        unit_results.append(
                            (
                                idx,
                                config_arg,
                                None,
                                AutotuneFilterReject(
                                    decision,
                                    filter_decisions=item["filter_decisions"],
                                ),
                            )
                        )
                        continue

                runtime_items.append(item)
            except Exception as e:
                unit_results.append((idx, config_arg, None, e))

        if not runtime_items:
            return unit_results

        runtime_grouped_config_indices = ",".join(str(item["idx"]) for item in runtime_items)
        runtime_group_size = len(runtime_items)
        with timed_autotune_stage(
            "grouped.merge_host_ir",
            group_size=runtime_group_size,
            configs=runtime_grouped_config_indices,
        ):
            merged_host_funcs: dict[Any, Any] = {}
            merged_host_attrs = None
            merged_host_names: set[str] = set()
            for item in runtime_items:
                host_mod = item["host_mod"]
                if merged_host_attrs is None:
                    merged_host_attrs = host_mod.attrs
                for global_var, func in host_mod.functions.items():
                    name_hint = getattr(global_var, "name_hint", str(global_var))
                    if name_hint in merged_host_names:
                        raise RuntimeError(
                            f"Duplicate host global symbol '{name_hint}' during grouped compilation (config index={item['idx']})."
                        )
                    merged_host_names.add(name_hint)
                    merged_host_funcs[global_var] = func
            merged_host_mod = tvm.IRModule(merged_host_funcs, attrs=merged_host_attrs)

        host_start = time.perf_counter()
        host_instruments, host_timing_inst = create_pass_instruments()
        with (
            timed_autotune_stage(
                "grouped.host_codegen",
                group_size=runtime_group_size,
                configs=runtime_grouped_config_indices,
            ),
            report_pass_timing_on_exit(
                host_timing_inst,
                context=f"stage=grouped-host, configs=[{runtime_grouped_config_indices}]",
            ),
            tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=host_instruments),
            runtime_items[0]["target"],
        ):
            grouped_host_rt_mod = host_codegen(
                merged_host_mod,
                runtime_items[0]["target_host"],
                target=runtime_items[0]["target"],
            )

        if carver_session is not None:
            duration = (time.perf_counter() - host_start) * 1000 / len(runtime_items)
            for item in runtime_items:
                carver_session.records[item["idx"]]["timings_ms"]["host_compile"] = duration

        with timed_autotune_stage(
            "grouped.import_module",
            group_size=runtime_group_size,
            configs=runtime_grouped_config_indices,
        ):
            grouped_host_rt_mod.import_module(grouped_device_rt_mod)

        shared_executable = tvm.runtime.Executable(grouped_host_rt_mod)
        with timed_autotune_stage(
            "grouped.executable_jit",
            group_size=runtime_group_size,
            configs=runtime_grouped_config_indices,
        ):
            shared_executable.jit()

        for item in runtime_items:
            idx = item["idx"]
            config_arg = item["config_arg"]
            try:
                kernel_symbol = str(item["program"].attrs["global_symbol"])
                artifact = CompiledArtifact(
                    host_mod=grouped_host_rt_mod,
                    device_mod=item["device_mod"],
                    params=item["params"],
                    kernel_source=grouped_kernel_source,
                    rt_mod=grouped_host_rt_mod,
                )

                with timed_autotune_stage(
                    "grouped.adapter_init",
                    group_size=lowered_group_size,
                    config_idx=idx,
                    configs=grouped_config_indices,
                ):
                    adapter = TVMFFIKernelAdapter(
                        params=artifact.params,
                        result_idx=compile_args.out_idx,
                        target=compile_args.target,
                        func_or_mod=item["program"],
                        host_mod=artifact.host_mod,
                        device_mod=artifact.device_mod,
                        rt_mod=artifact.rt_mod,
                        device_kernel_source=artifact.kernel_source,
                        entry_name=kernel_symbol,
                        executable=shared_executable,
                        verbose=compile_args.verbose,
                        pass_configs=pass_configs,
                    )
                    adapter._autotune_group_size = runtime_group_size
                    adapter._autotune_config_idx = idx

                with timed_autotune_stage(
                    "grouped.jit_kernel_init",
                    group_size=runtime_group_size,
                    config_idx=idx,
                    configs=runtime_grouped_config_indices,
                ):
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
                    jit_kernel._filter_decisions = item["filter_decisions"]

                unit_results.append((idx, config_arg, jit_kernel, None))
            except Exception as e:
                unit_results.append((idx, config_arg, None, e))
    except Exception as e:
        completed = {result[0] for result in unit_results}
        for item in lowered_items:
            if item["idx"] not in completed:
                unit_results.append((item["idx"], item["config_arg"], None, e))

    return unit_results

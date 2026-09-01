from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.autotuner.filters import (
    AutotuneFilterConfig,
    LaunchResourceInfo,
    evaluate_pre_compile_filter,
    extract_pre_compile_filter_info,
)
from tvm import tirx


def _evaluate(op_name, args):
    return tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get(op_name), args))


def _wgmma_call(op_name, c_data, *, shape="m64n128k16", c_dtype="float32", c_offset=0):
    a_data = tirx.Var("a_data", "handle")
    b_data = tirx.Var("b_data", "handle")
    e_data = tirx.Var("e_data", "handle")
    zero = tirx.IntImm("int32", 0)
    one = tirx.IntImm("int32", 1)
    true = tirx.const(True, "bool")
    shape_arg = tirx.StringImm(shape)
    f16 = tirx.StringImm("float16")
    c_dtype = tirx.StringImm(c_dtype)
    c_offset = tirx.IntImm("int32", c_offset)

    if op_name == "tl.ptx_wgmma_ss":
        args = [shape_arg, true, true, f16, f16, c_dtype, a_data, zero, b_data, zero, c_data, c_offset, one, true, true]
    elif op_name == "tl.ptx_wgmma_rs":
        args = [shape_arg, true, f16, f16, c_dtype, a_data, zero, b_data, zero, c_data, c_offset, one, true, true]
    elif op_name == "tl.ptx_wgmma_sp_ss":
        args = [
            shape_arg,
            true,
            true,
            f16,
            f16,
            c_dtype,
            a_data,
            zero,
            e_data,
            zero,
            zero,
            b_data,
            zero,
            c_data,
            c_offset,
            one,
            true,
            true,
        ]
    elif op_name == "tl.ptx_wgmma_sp_rs":
        args = [
            shape_arg,
            true,
            f16,
            f16,
            c_dtype,
            a_data,
            zero,
            e_data,
            zero,
            zero,
            b_data,
            zero,
            c_data,
            c_offset,
            one,
            true,
            true,
        ]
    else:
        raise ValueError(op_name)
    return _evaluate(op_name, args), a_data


def _fence(data, register_count, *, dtype="float32", offset=0):
    count = register_count if isinstance(register_count, tirx.PrimExpr) else tirx.IntImm("int32", register_count)
    return _evaluate(
        "tl.warpgroup_fence_operand",
        [tirx.StringImm(dtype), data, tirx.IntImm("int32", offset), count],
    )


def _set_max_nreg(register_count, is_increase):
    return _evaluate(
        "tl.set_max_nreg",
        [tirx.IntImm("int32", register_count), tirx.IntImm("int32", is_increase)],
    )


def _make_pressure_module(
    accumulator_registers=(64,),
    *,
    budget=240,
    op_names=None,
    shapes=None,
    fence_a_registers=None,
    duplicate_first_accumulator=False,
):
    op_names = op_names or ("tl.ptx_wgmma_ss",) * len(accumulator_registers)
    shapes = shapes or ("m64n128k16",) * len(accumulator_registers)
    assert len(op_names) == len(accumulator_registers)
    assert len(shapes) == len(accumulator_registers)

    consumer = [_set_max_nreg(budget, 1)]
    for index, (op_name, register_count, shape) in enumerate(zip(op_names, accumulator_registers, shapes)):
        c_data = tirx.Var(f"c_data_{index}", "handle")
        wgmma, a_data = _wgmma_call(op_name, c_data, shape=shape)
        if fence_a_registers is not None:
            consumer.append(_fence(a_data, fence_a_registers, dtype="float16"))
        consumer.extend([_fence(c_data, register_count), wgmma, _fence(c_data, register_count)])
        if duplicate_first_accumulator and index == 0:
            duplicate_wgmma, _ = _wgmma_call(op_name, c_data)
            consumer.append(duplicate_wgmma)

    tx = tirx.Var("tx", "int32")
    split = tirx.IfThenElse(
        tx < 128,
        _set_max_nreg(24, 0),
        tirx.SeqStmt(consumer),
    )
    body = tirx.AttrStmt(
        tvm.runtime.convert([128, 128]),
        "kWarpSpecializationScope",
        0,
        split,
    )
    func = tirx.PrimFunc([], body).with_attr("global_symbol", "main")
    return tvm.IRModule({"main": func})


@pytest.mark.parametrize(
    "op_name",
    [
        "tl.ptx_wgmma_ss",
        "tl.ptx_wgmma_rs",
        "tl.ptx_wgmma_sp_ss",
        "tl.ptx_wgmma_sp_rs",
    ],
)
def test_wgmma_pressure_reads_all_accumulator_operand_layouts(op_name):
    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=_make_pressure_module(op_names=(op_name,)),
        launch_info=LaunchResourceInfo("main", block_dims=(256, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.lower_bound_registers == 64
    assert pressure.upper_bound_registers == 64
    assert pressure.register_budget == 240
    assert pressure.status == "within_accumulator_budget"
    assert pressure.confidence == "exact"


def test_wgmma_pressure_bounds_multiple_storages_and_ignores_a_fence():
    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=_make_pressure_module(
            accumulator_registers=(64, 96),
            budget=128,
            op_names=("tl.ptx_wgmma_rs", "tl.ptx_wgmma_ss"),
            fence_a_registers=512,
        ),
        launch_info=LaunchResourceInfo("main", block_dims=(256, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.accumulator_storage_count == 2
    assert pressure.lower_bound_registers == 96
    assert pressure.upper_bound_registers == 160
    assert pressure.register_budget == 128
    assert pressure.headroom_lower_bound == -32
    assert pressure.headroom_upper_bound == 32
    assert pressure.status == "ambiguous"
    assert pressure.confidence == "bounded"


def test_wgmma_pressure_deduplicates_repeated_accumulator_storage():
    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=_make_pressure_module(duplicate_first_accumulator=True),
        launch_info=LaunchResourceInfo("main", block_dims=(256, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.accumulator_storage_count == 1
    assert pressure.lower_bound_registers == 64
    assert pressure.upper_bound_registers == 64
    assert pressure.wgmma_operation_count == 2


def test_wgmma_pressure_uses_maximum_across_mutually_exclusive_paths():
    c_data_0 = tirx.Var("c_data_0", "handle")
    c_data_1 = tirx.Var("c_data_1", "handle")
    wgmma_0, _ = _wgmma_call("tl.ptx_wgmma_ss", c_data_0)
    wgmma_1, _ = _wgmma_call("tl.ptx_wgmma_ss", c_data_1)
    predicate = tirx.Var("predicate", "bool")
    accumulator_choice = tirx.IfThenElse(
        predicate,
        tirx.SeqStmt([_fence(c_data_0, 64), wgmma_0, _fence(c_data_0, 64)]),
        tirx.SeqStmt([_fence(c_data_1, 96), wgmma_1, _fence(c_data_1, 96)]),
    )
    tx = tirx.Var("tx", "int32")
    split = tirx.IfThenElse(
        tx < 128,
        _set_max_nreg(24, 0),
        tirx.SeqStmt([_set_max_nreg(128, 1), accumulator_choice]),
    )
    body = tirx.AttrStmt(tvm.runtime.convert([128, 128]), "kWarpSpecializationScope", 0, split)
    device_mod = tvm.IRModule({"main": tirx.PrimFunc([], body).with_attr("global_symbol", "main")})

    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=device_mod,
        launch_info=LaunchResourceInfo("main", block_dims=(256, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.accumulator_storage_count == 2
    assert pressure.lower_bound_registers == 96
    assert pressure.upper_bound_registers == 96
    assert pressure.status == "within_accumulator_budget"
    assert pressure.confidence == "exact"


def test_wgmma_pressure_uses_static_allocation_as_conservative_upper_bound():
    c_data = tirx.Var(
        "c_data",
        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local"),
    )
    c_buffer = tirx.decl_buffer((96,), dtype="float32", data=c_data, scope="local")
    wgmma, _ = _wgmma_call("tl.ptx_wgmma_ss", c_data)
    tx = tirx.Var("tx", "int32")
    split = tirx.IfThenElse(
        tx < 128,
        _set_max_nreg(24, 0),
        tirx.SeqStmt(
            [
                _set_max_nreg(240, 1),
                tirx.AllocBuffer(c_buffer),
                _fence(c_data, 64),
                wgmma,
                _fence(c_data, 64),
            ]
        ),
    )
    body = tirx.AttrStmt(tvm.runtime.convert([128, 128]), "kWarpSpecializationScope", 0, split)
    device_mod = tvm.IRModule({"main": tirx.PrimFunc([], body).with_attr("global_symbol", "main")})

    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=device_mod,
        launch_info=LaunchResourceInfo("main", block_dims=(256, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.lower_bound_registers == 64
    assert pressure.upper_bound_registers == 96
    assert pressure.confidence == "bounded"


def test_wgmma_pressure_discards_fence_larger_than_static_allocation():
    c_data = tirx.Var(
        "c_data",
        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local"),
    )
    c_buffer = tirx.decl_buffer((128,), dtype="float32", data=c_data, scope="local")
    wgmma, _ = _wgmma_call(
        "tl.ptx_wgmma_ss",
        c_data,
        shape="m64n256k16",
        c_dtype="fp32",
    )
    tx = tirx.Var("tx", "int32")
    split = tirx.IfThenElse(
        tx < 128,
        _set_max_nreg(24, 0),
        tirx.SeqStmt(
            [
                _set_max_nreg(240, 1),
                tirx.AllocBuffer(c_buffer),
                _fence(c_data, 256),
                wgmma,
                _fence(c_data, 256),
            ]
        ),
    )
    body = tirx.AttrStmt(tvm.runtime.convert([128, 256]), "kWarpSpecializationScope", 0, split)
    device_mod = tvm.IRModule({"main": tirx.PrimFunc([], body).with_attr("global_symbol", "main")})

    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=device_mod,
        launch_info=LaunchResourceInfo("main", block_dims=(384, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.lower_bound_registers == 128
    assert pressure.upper_bound_registers == 128
    assert pressure.register_budget == 240
    assert pressure.status == "within_accumulator_budget"
    assert pressure.confidence == "lower_bound"
    assert "fence_exceeds_static_accumulator_allocation" in pressure.evidence
    assert "instruction_shape_fallback_used" in pressure.evidence


def test_wgmma_pressure_instruction_fallback_accounts_for_dtype_packing():
    c_data = tirx.Var("c_data", "handle")
    wgmma, _ = _wgmma_call("tl.ptx_wgmma_ss", c_data, c_dtype="float16")
    tx = tirx.Var("tx", "int32")
    split = tirx.IfThenElse(
        tx < 128,
        _set_max_nreg(24, 0),
        tirx.SeqStmt([_set_max_nreg(240, 1), wgmma]),
    )
    body = tirx.AttrStmt(tvm.runtime.convert([128, 128]), "kWarpSpecializationScope", 0, split)
    device_mod = tvm.IRModule({"main": tirx.PrimFunc([], body).with_attr("global_symbol", "main")})

    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=device_mod,
        launch_info=LaunchResourceInfo("main", block_dims=(256, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.lower_bound_registers == 32
    assert pressure.upper_bound_registers == 32
    assert pressure.confidence == "lower_bound"
    assert "instruction_shape_fallback_used" in pressure.evidence


def test_wgmma_pressure_over_budget_rejects_and_preserves_observation():
    device_mod = _make_pressure_module(accumulator_registers=(64,), budget=32)
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("main", block_dims=(256, 1, 1))],
        device_mod=device_mod,
        config={},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            action="reject",
            check_spills=False,
            check_local_memory=False,
            check_registers=False,
            check_c_local=False,
            check_output_elements_per_thread=False,
            check_k_loop=False,
            check_tma_tiny_tile=False,
        ),
    )

    assert decision.verdict == "reject"
    assert decision.reason == "filter_advisory_applied"
    assert decision.details["advisories"][0]["reason"] == "wgmma_register_pressure_over_budget"
    assert decision.details["advisories"][0]["observed"] == 64
    assert decision.details["advisories"][0]["limit"] == 32
    assert decision.details["observations"][0]["status"] == "over_budget"
    assert decision.details["observations"][0]["reason"] == "wgmma_register_pressure_observed"


def test_wgmma_within_budget_is_not_rejected_by_generic_accumulator_fallbacks():
    device_mod = _make_pressure_module(
        accumulator_registers=(128,),
        budget=240,
        shapes=("m64n256k16",),
    )
    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("main", block_dims=(256, 1, 1))],
        device_mod=device_mod,
        config={"block_M": 256, "block_N": 256, "thread_num": 128},
        filter_config=AutotuneFilterConfig(
            enabled=True,
            check_k_loop=False,
            check_tma_tiny_tile=False,
        ),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "filter_targets_passed"
    assert not decision.details["advisories"]
    assert decision.details["observations"][0]["status"] == "within_accumulator_budget"


def test_wgmma_pressure_reports_unknown_budget_without_warp_specialization():
    c_data = tirx.Var("c_data", "handle")
    wgmma, _ = _wgmma_call("tl.ptx_wgmma_ss", c_data)
    body = tirx.SeqStmt([_fence(c_data, 64), wgmma, _fence(c_data, 64)])
    func = tirx.PrimFunc([], body).with_attr("global_symbol", "main")
    device_mod = tvm.IRModule({"main": func})

    info = extract_pre_compile_filter_info(
        function_name="main",
        device_mod=device_mod,
        launch_info=LaunchResourceInfo("main", block_dims=(128, 1, 1)),
    )

    pressure = info.wgmma_register_pressure
    assert pressure is not None
    assert pressure.lower_bound_registers == 64
    assert pressure.upper_bound_registers == 64
    assert pressure.register_budget is None
    assert pressure.status == "budget_unknown"

    decision = evaluate_pre_compile_filter(
        launch_infos=[LaunchResourceInfo("main", block_dims=(128, 1, 1))],
        device_mod=device_mod,
        config={},
        filter_config=AutotuneFilterConfig(enabled=True),
    )

    assert decision.verdict == "keep"
    assert decision.reason == "filter_targets_passed"
    assert not decision.details["advisories"]
    assert decision.details["observations"][0]["status"] == "budget_unknown"

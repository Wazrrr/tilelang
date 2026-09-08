import pytest
import tilelang.language as T
from tvm import tirx as tir
from tvm.ir import Range
from tilelang.new_carver import analyze_prim_func, propagate_inputs


def gemm(trans_a=False, trans_b=False, explicit=False, stages=0, extent=4, input_dtype="float16", acc_dtype="float32", threads=128):
    @T.prim_func
    def main(A: T.Tensor((128, 128), input_dtype), B: T.Tensor((128, 128), input_dtype), C: T.Tensor((64, 64), acc_dtype)):
        with T.Kernel(1, threads=threads):
            a = T.alloc_shared((32, 32), input_dtype)
            b = T.alloc_shared((32, 32), input_dtype)
            c = T.alloc_fragment((32, 32), acc_dtype)
            if explicit:
                T.annotate_layout({c: T.Fragment((32, 32), forward_fn=lambda i, j: ((i * 32 + j) % 128, (i * 32 + j) // 128))})
            T.clear(c)
            for k in T.Pipelined(extent, num_stages=stages):
                T.copy(A[16:48, k * 32 : k * 32 + 32], a)
                T.copy(B[k * 32 : k * 32 + 32, 8:40], b)
                T.gemm(a, b, c, transpose_A=trans_a, transpose_B=trans_b)
            T.copy(c, C[8:40, 16:48])

    return main


@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (True, False), (False, True), (True, True)])
def test_gemm_regions(trans_a, trans_b):
    func = gemm(trans_a, trans_b)
    out = func.buffer_map[func.params[2]]
    demand = tir.BufferRegion(out, [Range.from_min_extent(12, 4), Range.from_min_extent(20, 8)])
    result = propagate_inputs(func, [demand])
    assert not result.unknown
    inputs = {r.buffer.name: r for r in result.per_iteration_inputs}
    assert set(inputs) == {"A", "B"}
    a = inputs["A"]
    b = inputs["B"]
    assert int(a.ranges[1 if trans_a else 0].extent) == 4
    assert int(b.ranges[0 if trans_b else 1].extent) == 8
    assert int(a.ranges[0 if trans_a else 1].extent) == 32
    assert any("k" in str(r.min) for r in a.ranges)
    coverage = {r.buffer.name: r for r in result.full_loop_inputs}
    assert any(int(r.extent) >= 100 for r in coverage["A"].ranges)
    assert any(op.dependencies for op in result.operations)


def test_unknown_budget_and_pipeline_do_not_reject():
    first = analyze_prim_func(gemm(stages=0, extent=1), True)
    second = analyze_prim_func(gemm(stages=3, extent=4), True)
    assert first["pressure"]["decision"]["keep"]
    assert second["pressure"]["decision"]["keep"]
    assert first["pressure"]["logical_storage"] == second["pressure"]["logical_storage"]
    assert first["pressure"]["total_register_upper_bound"] is None
    assert first["pressure"]["modeled_lower_bound"] == second["pressure"]["modeled_lower_bound"] == 8
    assert not analyze_prim_func(gemm(), {"register_cap": 1})["pressure"]["decision"]["keep"]


def test_analysis_resolves_register_budget_from_target():
    from tvm.target import Target

    func = gemm()
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    pressure = analyze_prim_func(func, target=target)["pressure"]
    assert pressure["budget"] == 255
    assert pressure["budget_source"] == "architecture register limit"
    assert pressure["hardware_register_cap"] == 255
    assert pressure["target_arch"] == "sm_90a"
    assert pressure["decision"]["keep"]
    attributed = func.with_attr("target", target)
    assert analyze_prim_func(attributed)["pressure"] == pressure
    with target:
        assert analyze_prim_func(func)["pressure"] == pressure
    assert analyze_prim_func(attributed, target="llvm")["pressure"]["budget"] is None
    assert analyze_prim_func(func, {"register_cap": 4}, target=target)["pressure"]["decision"]["would_reject"]


@pytest.mark.parametrize(
    "input_dtype,acc_dtype,bound", [("float16", "float16", 4), ("float16", "float32", 8), ("int8", "int32", 8), ("float64", "float64", 16)]
)
def test_automatic_layout_dtype_bound(input_dtype, acc_dtype, bound):
    func = gemm(input_dtype=input_dtype, acc_dtype=acc_dtype)
    pressure = analyze_prim_func(func, {"register_cap": bound - 1})["pressure"]
    assert pressure["modeled_lower_bound"] == bound
    assert not pressure["decision"]["keep"]
    assert pressure["total_register_upper_bound"] is None
    assert any("no balanced mapping assumed" in item for item in pressure["evidence"])
    assert analyze_prim_func(func, {"register_cap": bound})["pressure"]["decision"]["keep"]
    report = analyze_prim_func(func, {"register_cap": bound - 1, "mode": "report_only"})["pressure"]
    assert report["decision"]["keep"] and report["decision"]["would_reject"]


@pytest.mark.parametrize("threads,bound", [(96, 6), ((32, 4), 4), (256, 2)])
def test_automatic_layout_thread_bound_and_rounding(threads, bound):
    pressure = analyze_prim_func(gemm(acc_dtype="float16", threads=threads))["pressure"]
    assert pressure["modeled_lower_bound"] == bound


def test_unknown_thread_extent_keeps_automatic_layout_eligible():
    def symbolic_threads(node):
        if isinstance(node, tir.For) and node.thread_binding is not None and node.thread_binding.thread_tag == "threadIdx.x":
            return tir.For(node.loop_var, node.min, tir.Var("unknown_threads", "int32"), node.kind, node.body, node.thread_binding)

    func = gemm()
    func = func.with_body(tir.stmt_functor.ir_transform(func.body, None, symbolic_threads))
    pressure = analyze_prim_func(func, {"register_cap": 1})["pressure"]
    assert pressure["modeled_lower_bound"] is None
    assert pressure["decision"]["keep"]


@pytest.mark.parametrize("uncertainty", ["predicate", "opaque", "missing_threads"])
def test_automatic_gemm_unknown_cases_remain_eligible(uncertainty):
    func = gemm()
    if uncertainty == "predicate":
        body = tir.IfThenElse(tir.Var("condition", "bool"), func.body, None)
    elif uncertainty == "opaque":
        body = tir.SeqStmt([func.body, tir.Evaluate(tir.call_extern("int32", "opaque"))])
    else:

        def remove_threads(node):
            if isinstance(node, tir.For) and node.thread_binding is not None:
                return tir.For(node.loop_var, node.min, node.extent, tir.ForKind.SERIAL, node.body)

        body = tir.stmt_functor.ir_transform(func.body, None, remove_threads)
    pressure = analyze_prim_func(func.with_body(body), {"register_cap": 1})["pressure"]
    assert pressure["modeled_lower_bound"] is None
    assert pressure["decision"]["keep"]


def test_tensor_memory_accumulator_is_not_register_storage():
    @T.prim_func
    def main(A: T.Tensor((128, 32), "float16"), B: T.Tensor((32, 128), "float16"), C: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 32), "float16")
            b = T.alloc_shared((32, 128), "float16")
            c = T.alloc_tmem((128, 128), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.tcgen05_gemm(a, b, c, clear_accum=True, mbar=None)
            T.copy(c, C)

    result = analyze_prim_func(main, {"register_cap": 1})
    assert not result["tile_propagation"]["unknown"]
    assert result["pressure"]["modeled_lower_bound"] is None
    assert result["pressure"]["decision"]["keep"]


def test_explicit_layout_pressure():
    pressure = analyze_prim_func(gemm(explicit=True), {"register_cap": 1})["pressure"]
    c = next(x for x in pressure["logical_storage"] if x["buffer"] == "c")
    assert c["modeled_registers_per_thread"] == {"lower": 8, "upper": 8}
    assert pressure["modeled_lower_bound"] == 8
    assert not pressure["decision"]["keep"]
    assert analyze_prim_func(gemm(explicit=True), {"register_cap": 1, "mode": "report_only"})["pressure"]["decision"]["keep"]


def test_overwrite_and_broadcast():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1, threads=32):
            tmp = T.alloc_fragment((16,), "float32")
            T.copy(A, tmp)
            T.copy(B, tmp)
            for i, j in T.Parallel(16, 16):
                C[i, j] = T.cast(tmp[j], "float32")

    output = main.buffer_map[main.params[2]]
    result = propagate_inputs(main, [tir.BufferRegion(output, [Range.from_min_extent(0, 16), Range.from_min_extent(0, 16)])])
    assert {r.buffer.name for r in result.per_iteration_inputs} == {"B"}


def test_branches_and_opaque_remain_eligible():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            tmp = T.alloc_fragment((16,), "float32")
            if bx == 0:
                T.copy(A, tmp)
            else:
                T.copy(B, tmp)
            T.copy(tmp, C)
            T.evaluate(T.call_extern("int32", "opaque"))

    result = analyze_prim_func(main, {"register_cap": 1})
    assert result["pressure"]["decision"]["keep"]
    assert result["tile_propagation"]["unknown"]
    assert {r["buffer"] for r in result["tile_propagation"]["per_iteration_inputs"]} == {"A", "B"}


def test_reduce():
    @T.prim_func
    def main(A: T.Tensor((16, 32), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((16, 32), "float32")
            dst = T.alloc_fragment((16,), "float32")
            T.copy(A, src)
            T.reduce_sum(src, dst, dim=1)
            T.copy(dst, C)

    c = main.buffer_map[main.params[1]]
    result = propagate_inputs(main, [tir.BufferRegion(c, [Range.from_min_extent(3, 4)])])
    a = result.per_iteration_inputs[0]
    assert int(a.ranges[0].min) == 3
    assert [int(r.extent) for r in a.ranges] == [4, 32]


def test_boundary_copy():
    @T.prim_func
    def main(A: T.Tensor((19,), "float32"), C: T.Tensor((19,), "float32")):
        with T.Kernel(1, threads=32):
            tmp = T.alloc_fragment((16,), "float32")
            T.copy(A[16:32], tmp)
            T.copy(tmp, C[0:16])

    output = main.buffer_map[main.params[1]]
    result = propagate_inputs(main, [tir.BufferRegion(output, [Range.from_min_extent(0, 16)])])
    assert int(result.per_iteration_inputs[0].ranges[0].extent) == 3
    assert result.per_iteration_inputs[0].precision == "conservative"


@pytest.mark.parametrize("dtype,bounds", [("float16", {"lower": 4, "upper": 8}), ("float32", {"lower": 8, "upper": 8})])
def test_packing_and_register_operands(dtype, bounds):
    @T.prim_func
    def main(A: T.Tensor((32, 32), dtype), C: T.Tensor((32, 32), dtype)):
        with T.Kernel(1, threads=128):
            tmp = T.alloc_fragment((32, 32), dtype)
            T.annotate_layout({tmp: T.Fragment((32, 32), forward_fn=lambda i, j: ((i * 32 + j) % 128, (i * 32 + j) // 128))})
            T.copy(A, tmp)
            T.copy(tmp, C)

    pressure = analyze_prim_func(main, {"register_cap": 1})["pressure"]
    tmp = next(x for x in pressure["logical_storage"] if x["buffer"] == "tmp")
    assert tmp["modeled_registers_per_thread"] == bounds
    assert pressure["decision"]["keep"]  # copy can stream/fuse
    assert pressure["live_tile_sets"]


def test_partial_overwrite_keeps_unwritten_inputs():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((8,), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            tmp = T.alloc_fragment((16,), "float32")
            T.copy(A, tmp)
            T.copy(B, tmp[8:16])
            T.copy(tmp, C)

    output = main.buffer_map[main.params[2]]
    result = propagate_inputs(main, [tir.BufferRegion(output, [Range.from_min_extent(0, 16)])])
    assert {r.buffer.name for r in result.per_iteration_inputs} == {"A", "B"}


def test_strided_writes_do_not_kill_unwritten_values():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((8,), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            tmp = T.alloc_fragment((16,), "float32")
            T.copy(A, tmp)
            for i in T.Parallel(8):
                tmp[2 * i] = B[i]
            T.copy(tmp, C)

    output = main.buffer_map[main.params[2]]
    result = propagate_inputs(main, [tir.BufferRegion(output, [Range.from_min_extent(0, 16)])])
    assert {r.buffer.name for r in result.per_iteration_inputs} == {"A", "B"}


def test_global_initialization_is_not_an_external_input():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), C: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            T.fill(A, 0)
            T.copy(A, C)

    output = main.buffer_map[main.params[1]]
    assert not propagate_inputs(main, [tir.BufferRegion(output, [Range.from_min_extent(0, 16)])]).per_iteration_inputs


def test_explicit_replication():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float16"), B: T.Tensor((32, 32), "float16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, threads=256):
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            c = T.alloc_fragment((32, 32), "float32")
            T.annotate_layout({c: T.Fragment((32, 32), forward_fn=lambda i, j: ((i * 32 + j) % 128, (i * 32 + j) // 128)).replicate(2)})
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            T.gemm(a, b, c)
            T.copy(c, C)

    pressure = analyze_prim_func(main, {"register_cap": 4})["pressure"]
    c = next(x for x in pressure["logical_storage"] if x["buffer"] == "c")
    assert c["replication"] == 2
    assert c["computing_threads"] == 256
    assert pressure["modeled_lower_bound"] == 8


def test_region_query_is_separate_from_kernel_pressure():
    func = gemm(explicit=True)
    out = func.buffer_map[func.params[2]]
    demand = tir.BufferRegion(out, [Range.from_min_extent(8, 1), Range.from_min_extent(16, 1)])
    before = analyze_prim_func(func, {"register_cap": 1})["pressure"]
    query = propagate_inputs(func, [demand])
    assert all(any(int(r.extent) == 1 for r in region.ranges) for region in query.per_iteration_inputs)
    assert analyze_prim_func(func, {"register_cap": 1})["pressure"] == before
    with pytest.raises(TypeError, match="outputs"):
        analyze_prim_func(func, outputs=[demand])


@pytest.mark.parametrize("outputs,error", [(None, TypeError), ([], ValueError)])
def test_region_query_requires_outputs(outputs, error):
    with pytest.raises(error, match="output"):
        propagate_inputs(gemm(), outputs)


def test_analysis_requires_global_output():
    @T.prim_func
    def main(A: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            tmp = T.alloc_fragment((16,), "float32")
            T.copy(A, tmp)

    with pytest.raises(ValueError, match="global output write"):
        analyze_prim_func(main)

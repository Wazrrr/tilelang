"""Reuse must respect loop iterations, simultaneous operands and explicit policy."""

import tilelang.language as T
from tilelang.new_carver.analysis import _Collector, _kernel_outputs, _propagate_tiles
from tilelang.new_carver.memory import analyze_memory
from tilelang.new_carver.shared_storage import shared_storage_plan


def plan(func, **passes):
    col = _Collector(func)
    memory = analyze_memory(col, _propagate_tiles(col, _kernel_outputs(col)))
    return shared_storage_plan(col, memory["shared_allocations"], passes)


def test_disjoint_tiles_share_storage_without_buffer_name_rules():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            x = T.alloc_shared((128,), "float32")
            y = T.alloc_shared((128,), "float32")
            z = T.alloc_fragment((128,), "float32")
            T.copy(A, x)
            T.copy(x, z)
            T.copy(z, y)
            T.copy(y, B)

    result = plan(main)
    assert result["allocated_sum_bytes"] == 1024
    assert result["arena_bytes_estimate"] == 512
    assert plan(main, **{"tl.disable_shared_memory_reuse": True})["arena_bytes_estimate"] == 1024


def test_same_operation_operands_cannot_alias():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            x = T.alloc_shared((128,), "float32")
            y = T.alloc_shared((128,), "float32")
            T.copy(A, x)
            T.copy(x, y)
            T.copy(y, B)

    assert plan(main)["arena_bytes_estimate"] == 1024


def test_pipeline_lifetimes_expand_even_when_lexical_uses_are_disjoint():
    @T.prim_func
    def main(A: T.Tensor((4, 128), "float32"), B: T.Tensor((4, 128), "float32")):
        with T.Kernel(1, threads=128):
            x = T.alloc_shared((128,), "float32")
            y = T.alloc_shared((128,), "float32")
            z = T.alloc_fragment((128,), "float32")
            for k in T.Pipelined(4, num_stages=3):
                T.copy(A[k, :], x)
                T.copy(x, z)
                T.copy(z, y)
                T.copy(y, B[k, :])

    result = plan(main)
    assert result["arena_bytes_estimate"] == 2 * 3 * 512
    assert len({(i["start"], i["end"]) for i in result["intervals"]}) == 1


def test_real_attention_epilogue_reuses_an_input_tile():
    from examples.flash_attention.example_mha_new_carver import make_attention

    func = make_attention(dim=256)(128, 128, 1, 256)
    before = func.script()
    result = plan(func)
    assert result["allocated_sum_bytes"] == 262144
    assert result["arena_bytes_estimate"] == 196608
    assert result["reuse_bytes_estimate"] == 65536
    assert func.script() == before


def test_opaque_uses_never_enable_reuse():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            x = T.alloc_shared((128,), "float32")
            y = T.alloc_shared((128,), "float32")
            T.copy(A, x)
            T.evaluate(T.call_extern("int32", "opaque_shared_operation", x.data, y.data))
            T.copy(y, B)

    result = plan(main)
    assert result["precision"] == "unknown"
    assert result["arena_bytes_estimate"] == result["allocated_sum_bytes"] == 1024


def test_shared_reuse_honors_function_and_call_pass_settings():
    from examples.flash_attention.example_mha_new_carver import make_attention
    from tilelang.new_carver import analyze_prim_func
    from test_cost import LIMITS
    from test_modules import TARGET

    func = make_attention(dim=256)(128, 128, 1, 256).with_attr("tilelang_pass_configs", {"tl.disable_shared_memory_reuse": True})
    blocked = analyze_prim_func(func, target=TARGET, device_limits=LIMITS)
    allowed = analyze_prim_func(func, target=TARGET, device_limits=LIMITS, pass_configs={"tl.disable_shared_memory_reuse": False})
    assert blocked["modules"]["memory_traffic"]["shared_memory_bytes_estimate"] == 262144
    assert allowed["modules"]["memory_traffic"]["shared_memory_bytes_estimate"] == 196608
    assert allowed["modules"]["waves"]["resident_blocks_per_sm_estimate"] == 1
    for result in (blocked, allowed):
        result["pressure"]["warp_specialization"].pop("effective_pass_configs")
    assert blocked["pressure"] == allowed["pressure"]

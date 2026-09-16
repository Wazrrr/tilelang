"""Optimized cardinalities must equal the bounded ownership reference."""

import pytest
from regression_kernels import softmax_program
from tilelang.tiletune.ampere import prepare_analysis, _PREPARATION_CACHE
from tilelang.tiletune.compute import scalar_fragment_work
from tilelang.tiletune.src.collector import _Collector
from test_portability import AMPERE


@pytest.mark.parametrize("rows,cols,threads,vector,row_threads", [(4, 128, 128, 1, 4), (8, 512, 256, 4, 4), (4, 1024, 128, 2, 1)])
def test_exact_projection_counts_match_reference(rows, cols, threads, vector, row_threads):
    func = softmax_program(32, 2048, "float16", rows, cols, threads, vector, row_threads)
    col = _Collector(func)
    prepare_analysis(func, col, AMPERE, {})
    for op in col.operations:
        if op.kind == "elementwise":
            layout = col.inferred_layouts[op.metadata.buffer.data]
            assert scalar_fragment_work(op, layout) == scalar_fragment_work(op, layout, reference=True)


def test_cache_keeps_reports_independent_and_invalidates_pass_settings():
    _PREPARATION_CACHE.clear()
    func = softmax_program(32, 256, "float16", 4, 128, 128, 1, 4)
    before = func.script()
    first, second = _Collector(func), _Collector(func)
    prepare_analysis(func, first, AMPERE, {})
    first.ampere_plan["unknown"].append("mutated report")
    prepare_analysis(func, second, AMPERE, {})
    assert not second.ampere_plan["unknown"]
    assert len(_PREPARATION_CACHE) == 1
    prepare_analysis(func, _Collector(func), AMPERE, {"tl.disable_shared_memory_reuse": True})
    assert len(_PREPARATION_CACHE) == 2
    assert func.script() == before
    assert all(buffer.data in second.inferred_layouts for buffer in second.buffers if buffer.data in first.inferred_layouts)


def test_explicit_fast_path_matches_full_native_inference(monkeypatch):
    from tilelang.tiletune import ownership
    from test_ampere import analyze

    func = softmax_program(31, 259, "float16", 4, 128, 128, 1, 4)
    _PREPARATION_CACHE.clear()
    fast = analyze(func)
    _PREPARATION_CACHE.clear()
    monkeypatch.setattr(ownership, "verified_explicit_layouts", lambda *args: None)
    reference = analyze(func)
    assert fast["tile_cost"]["score"] == reference["tile_cost"]["score"]
    assert fast["diagnostics"] == reference["diagnostics"]
    for x, y in zip(fast["modules"]["pipeline_overlap"]["phases"], reference["modules"]["pipeline_overlap"]["phases"]):
        assert x["work"] == y["work"]
        assert x["reduction"] == y["reduction"]


@pytest.mark.parametrize("stages,block_n", [(0, 32), (2, 64), (3, 128)])
def test_mixed_mma_explicit_ownership_matches_native(monkeypatch, stages, block_n):
    from regression_kernels import attention_program
    from tilelang.tiletune import ownership
    from test_ampere import analyze

    func = attention_program(1, 2, 129, 64, True, "float16", 32, block_n, stages, 128, "square", "square", None)
    before = func.script()
    _PREPARATION_CACHE.clear()
    fast = analyze(func)
    _PREPARATION_CACHE.clear()
    monkeypatch.setattr(ownership, "verified_explicit_layouts", lambda *args: None)
    reference = analyze(func)
    assert func.script() == before
    assert fast["tile_cost"]["score"] == reference["tile_cost"]["score"]
    assert fast["diagnostics"] == reference["diagnostics"]
    for x, y in zip(fast["modules"]["pipeline_overlap"]["phases"], reference["modules"]["pipeline_overlap"]["phases"]):
        assert x["work"] == y["work"]
        assert x["reduction"] == y["reduction"]


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_shared_buffer_versions_match_native_pipeline(stages):
    from regression_kernels import attention_program
    from tilelang import tvm, transform
    from tvm import tirx as tir
    from tvm.target import Target
    from tilelang.tiletune.shared_memory import analyze_shared_memory
    from tilelang.tiletune.src.buffer_facts import collect_buffer_facts

    func = attention_program(1, 2, 256, 64, True, "float16", 32, 64, stages, 128, "square", "square", None)
    col = _Collector(func)
    prepare_analysis(func, col, AMPERE, {})
    predicted = analyze_shared_memory(col, collect_buffer_facts(col))
    mod = tir.transform.BindTarget(Target(AMPERE))(tvm.IRModule({"main": func}))
    for factory in (transform.IfStmtBinding, transform.PipelinePlanning, transform.InjectSoftwarePipeline):
        mod = factory()(mod)
    allocated = {}

    def visit(node):
        if isinstance(node, tir.AllocBuffer) and node.buffer.scope().startswith("shared"):
            allocated[node.buffer.name] = node.buffer
        if isinstance(node, tir.SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope().startswith("shared"):
                    allocated[buffer.name] = buffer

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    from math import prod

    for entry in predicted["shared_allocations"]:
        buffer = allocated[entry["buffer"]]
        actual = prod(int(n) for n in buffer.shape) * tvm.DataType(buffer.dtype).bits // 8
        assert entry["allocated_bytes_estimate"] == actual
    copies = {e["buffer"]: e["pipeline_copies_estimate"] for e in predicted["shared_allocations"]}
    assert copies["k"] == copies["v"] == stages
    assert copies["probabilities"] == copies["scores_shared"] == copies["rescale_shared"] == 1


def test_cache_context_keys_preserve_value_types_and_mapping_order():
    from tilelang.tiletune.src.structural_key import context_key

    assert context_key({"a": 1, "b": [True]}) == context_key({"b": [True], "a": 1})
    assert context_key({"a": 1}) != context_key({"a": "1"})
    assert context_key({"a": 1}) != context_key({"a": True})
    assert context_key({"a": [1]}) != context_key({"a": "[1]"})


def test_cached_explicit_layouts_use_fresh_buffer_identities(monkeypatch):
    from tilelang.tiletune import ampere

    first = softmax_program(31, 259, "float16", 4, 128, 128, 1, 4)
    second = softmax_program(31, 259, "float16", 4, 128, 128, 1, 4)
    before, after = _Collector(first), _Collector(second)
    _PREPARATION_CACHE.clear()
    prepare_analysis(first, before, AMPERE, {})
    assert before.inferred_layouts

    def unexpected_preparation(*args):
        pytest.fail("equivalent explicit layouts should reuse cached preparation")

    monkeypatch.setattr(ampere, "_prepare_analysis", unexpected_preparation)
    prepare_analysis(second, after, AMPERE, {})
    assert set(before.inferred_layouts).isdisjoint(after.inferred_layouts)
    assert len(before.inferred_layouts) == len(after.inferred_layouts)
    for old, new in zip(before.operations, after.operations):
        if old.kind == "elementwise":
            assert scalar_fragment_work(old, before.inferred_layouts[old.metadata.buffer.data]) == scalar_fragment_work(
                new, after.inferred_layouts[new.metadata.buffer.data]
            )


@pytest.mark.parametrize("stages,intra_stages", [(0, 0), (2, 3)])
def test_sequential_kda_cache_rebinds_fresh_equivalent_ir(monkeypatch, stages, intra_stages):
    from regression_kernels import chunk_program
    from tilelang.tiletune import ampere
    from test_ampere import analyze

    args = (1, 2, 256, 96, 64, 128, "float16", 32, 32, stages, 128, 32, 32, intra_stages)
    func = chunk_program(*args)
    equivalent = chunk_program(*args)
    assert not func.same_as(equivalent)
    before = func.script()
    _PREPARATION_CACHE.clear()
    reference = analyze(func)

    def unexpected_preparation(*args):
        pytest.fail("equivalent KDA IR should reuse the cached compiler plan")

    monkeypatch.setattr(ampere, "_prepare_analysis", unexpected_preparation)
    fast = analyze(equivalent)
    assert func.script() == before
    assert equivalent.script() == before
    assert fast["tile_cost"]["score"] is not None
    assert fast["tile_cost"]["score"] == reference["tile_cost"]["score"]
    assert fast["diagnostics"] == reference["diagnostics"]
    assert fast["pressure"]["decision"] == reference["pressure"]["decision"]
    for x, y in zip(fast["modules"]["pipeline_overlap"]["phases"], reference["modules"]["pipeline_overlap"]["phases"]):
        assert x["work"] == y["work"]
        assert x["reduction"] == y["reduction"]

"""One tile graph supplies pressure evidence, address coverage and memory cost."""

import tilelang.language as T
from tvm import tirx as tir
from tvm.arith import Analyzer
from tilelang.tiletune import analyze_prim_func
from tilelang.tiletune.analysis import _Collector, _kernel_outputs, _propagate_tiles


def tiled_gemm(repeated=False):
    @T.prim_func
    def main(A: T.Tensor((96, 128), "float16"), B: T.Tensor((128, 64), "float16"), C: T.Tensor((96, 64), "float32")):
        with T.Kernel(3, 2, threads=128) as (bx, by):
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)
            for ki in T.Pipelined(4, num_stages=0):
                T.copy(A[bx * 32, 0 if repeated else ki * 32], a)
                T.copy(B[0 if repeated else ki * 32, by * 32], b)
                T.gemm(a, b, c)
            # Scalar output stores must describe the same complete CTA tile.
            for i, j in T.Parallel(32, 32):
                C[bx * 32 + i, by * 32 + j] = c[i, j]

    return main


def test_one_tile_traversal_preserves_accumulator_proof(monkeypatch):
    from tilelang.tiletune import analysis

    calls = []
    propagate = analysis._propagate_tiles

    def observe(col, outputs):
        result = propagate(col, outputs)
        calls.append(result)
        return result

    monkeypatch.setattr(analysis, "_propagate_tiles", observe)
    result = analyze_prim_func(tiled_gemm(), {"register_cap": 7})
    assert len(calls) == 1
    assert "propagation" not in result
    assert result["tile_propagation"] == calls[0].to_dict()
    assert result["pressure"]["modeled_lower_bound"] == 8
    assert result["pressure"]["decision"]["would_reject"]
    gemm = next(op for op in calls[0].operations if op.kind == "gemm")
    assert any([int(r.extent) for r in demand.ranges] == [32, 32] for demand in gemm.demands)
    iteration = {r.buffer.name: r for r in calls[0].per_iteration_inputs}
    coverage = {r.buffer.name: r for r in calls[0].full_loop_inputs}
    assert [int(r.extent) for r in iteration["A"].ranges] == [32, 32]
    assert [int(r.extent) for r in coverage["A"].ranges] == [32, 128]
    assert [int(r.extent) for r in coverage["B"].ranges] == [128, 32]
    assert "bx" in str(coverage["A"].ranges[0].min)
    assert "by" in str(coverage["B"].ranges[1].min)


def test_repeated_accesses_have_traffic_without_larger_coverage():
    result = analyze_prim_func(tiled_gemm(repeated=True))
    memory = result["modules"]["memory_traffic"]
    assert memory["input_bytes_per_block"] == 4 * (32 * 32 * 2 + 32 * 32 * 2)
    assert memory["output_bytes_per_block"] == 32 * 32 * 4
    assert all(tile["visits_per_block"] == 4 for tile in memory["input_tiles"])
    for region in result["tile_propagation"]["full_loop_inputs"]:
        assert [r["extent"] for r in region["ranges"]] == ["32", "32"]
    assert result["pressure"]["modeled_lower_bound"] == 8


def test_boundary_tile_keeps_its_symbolic_address_and_valid_extent():
    @T.prim_func
    def main(A: T.Tensor((70,), "float32"), C: T.Tensor((70,), "float32")):
        with T.Kernel(3, threads=32) as bx:
            tile = T.alloc_fragment((32,), "float32")
            T.copy(A[bx * 32], tile)
            T.copy(tile, C[bx * 32])

    col = _Collector(main)
    result = _propagate_tiles(col, _kernel_outputs(col))
    bx, _ = col.block_domains["blockIdx.x"]
    for region in (result.per_iteration_inputs[0], result.full_loop_inputs[0]):
        axis = region.ranges[0]
        last = {bx: tir.IntImm("int32", 2)}
        assert int(Analyzer().simplify(tir.stmt_functor.substitute(axis.min, last))) == 64
        assert int(Analyzer().simplify(tir.stmt_functor.substitute(axis.extent, last))) == 6
        assert "bx" in str(axis.min)
        assert region.precision == "conservative"

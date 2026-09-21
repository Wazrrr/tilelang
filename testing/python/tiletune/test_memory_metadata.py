"""Memory scoring resolves needed metadata without rewriting executable IR."""

import pytest
import tilelang.language as T

from tilelang.tiletune import TileTuneConfig, analyze_prim_func
from test_cost import LIMITS
from test_modules import TARGET


def compare_modes(func, values, **settings):
    before = func.script()
    lean, detailed = [
        analyze_prim_func(
            func,
            dict(ranking_metric="memory", memory_diagnostics=diagnostics, input_values=values, **settings),
            target=TARGET,
            device_limits=LIMITS,
        )
        for diagnostics in (False, True)
    ]
    assert func.script() == before
    for key in (
        "score",
        "logical_byte_waves",
        "logical_memory_access_waves",
        "grid_blocks",
        "pipeline_depth",
        "precision",
        "unknown",
    ):
        assert lean["tile_cost"][key] == detailed["tile_cost"][key], key
    for key in ("keep", "would_reject", "physical_reasons", "policy_reasons"):
        assert lean["pressure"]["decision"][key] == detailed["pressure"]["decision"][key], key
    fields = ("operation", "direction", "buffer", "bytes", "visits", "visit_precision", "predicated")
    assert [[access[key] for key in fields] for access in lean["tile_cost"]["accesses"]] == [
        [access[key] for key in fields] for access in detailed["tile_cost"]["accesses"]
    ]
    return lean


@pytest.mark.parametrize("sizes", [[64, 128], [63, 77, 111, 280]])
@pytest.mark.parametrize("stages", [0, 3])
@pytest.mark.parametrize("strict", [False, True])
def test_grouped_metadata_preserves_accesses_and_resource_decisions(sizes, stages, strict):
    from experiments.common.spec import Workload
    from experiments.grouped_gemm.kernel import make_case

    case = make_case(Workload("metadata", "grouped_gemm", dict(batch_sizes=sizes, n=127, k=95)))
    func = case.build(block_M=64, block_N=64, block_K=32, num_stages=stages, threads=128)
    settings = dict(register_cap=1) if strict else dict(max_spill_bytes=None, max_local_bytes=None)
    result = compare_modes(func, case.input_values, **settings)
    assert result["ir_context"]["metadata_resolution"] == "deferred"
    assert result["tile_cost"]["score"] is not None
    metadata = [access for access in result["tile_cost"]["accesses"] if access["buffer"] not in ("A", "B", "C")]
    assert sum(access["bytes"] * access["visits"] for access in metadata) == (len(sizes) + 4) * 4


@pytest.mark.parametrize("sizes,visits", [([2, 4], 4), ([3, 3], 3)])
def test_metadata_driven_loop_bounds_are_resolved(sizes, visits):
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float32"), Sizes: T.Tensor((2,), "int32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            count = Sizes[bx]
            for i in T.serial(count):
                B[bx * 8 + i] = A[bx * 8 + i]

    result = compare_modes(kernel, {"1": sizes})
    assert result["tile_cost"]["logical_byte_waves"] == 4 + 8 * visits
    assert result["tile_cost"]["logical_memory_access_waves"] == 1 + 2 * visits


@pytest.mark.parametrize("sizes", [[3, 7], [6, 2]])
def test_native_access_extents_use_metadata_and_launch_bounds(sizes):
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float32"), Sizes: T.Tensor((2,), "int32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            count = T.if_then_else(bx < 2, Sizes[0], Sizes[1])
            tile = T.alloc_shared((8,), "float32")
            T.copy(A[bx * 8 : bx * 8 + count], tile[:count])
            T.copy(tile[:count], B[bx * 8 : bx * 8 + count])

    result = compare_modes(kernel, {"1": sizes})
    assert result["tile_cost"]["logical_byte_waves"] == 8 + 8 * sizes[0]
    assert result["ir_context"]["metadata_resolution"] == "deferred"


def test_unreachable_opaque_binding_retries_eager_resolution():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float32"), Offsets: T.Tensor((2,), "int32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            offset = T.if_then_else(bx < 2, Offsets[bx], T.call_extern("int32", "opaque_index", A.data))
            B[bx] = A[offset]

    result = compare_modes(kernel, {"1": [0, 8]})
    assert result["tile_cost"]["score"] is not None
    assert result["ir_context"]["metadata_resolution"] == "eager"


def test_out_of_range_metadata_preserves_known_volume_without_guessing_address():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float32"), Offsets: T.Tensor((2,), "int32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            offset = Offsets[bx + 1]
            B[bx] = A[offset]

    result = compare_modes(kernel, {"1": [0, 8]})
    # B200 keeps address uncertainty separate from volume uncertainty. The
    # scalar access still has a proven size and visit count, so it remains
    # scoreable without pretending the out-of-range lookup was resolved.
    assert result["tile_cost"]["score"] is not None
    assert result["ir_context"]["metadata_resolution"] == "eager"


def test_memory_metadata_remains_read_only():
    @T.prim_func
    def kernel(Sizes: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=32):
            Sizes[0] = 1

    with pytest.raises(ValueError, match="read-only"):
        analyze_prim_func(kernel, dict(ranking_metric="memory", input_values={"0": [1, 2]}), target=TARGET)


@pytest.mark.parametrize("values", [{"-1": [1]}, {"x": [1]}, {"1": []}, {"1": [True]}, {"1": [1.5]}])
def test_metadata_contract_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="input_values"):
        TileTuneConfig(input_values=values)


@pytest.mark.parametrize("values", [{"0": [1] * 16}, {"1": [1]}, {"9": [1]}])
def test_metadata_contract_checks_parameter_dtype_shape_and_index(values):
    @T.prim_func
    def kernel(A: T.Tensor((16,), "float32"), Sizes: T.Tensor((2,), "int32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            B[0] = A[Sizes[0]]

    with pytest.raises(ValueError, match="input_values"):
        analyze_prim_func(kernel, dict(ranking_metric="memory", input_values=values), target=TARGET)

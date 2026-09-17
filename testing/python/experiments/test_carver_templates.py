"""New templates preserve the mathematical graph and use the existing policy."""

import json
import pytest
from experiments.common.baselines import carver_rank
from experiments.common.spec import Device, TARGETS
from experiments.families import family_module


@pytest.fixture
def offline_arch(monkeypatch):
    from tilelang import tvm
    from tilelang.carver.arch import CUDA
    from tilelang.carver import arch as arch_module

    # Fixed policy inputs for CPU-only regression tests; no hardware timing.
    arch = CUDA.__new__(CUDA)
    arch.__dict__.update(
        target=tvm.target.Target(dict(kind="cuda", arch="sm_90")),
        sm_version=90,
        name="test",
        platform="CUDA",
        smem_cap=49152,
        compute_max_core=132,
        warp_size=32,
        compute_capability="90",
        reg_cap=65536,
        max_smem_usage=98304,
        sm_partition=4,
        l2_cache_size_bytes=52428800,
        transaction_size=[32, 128],
        bandwidth=[750, 12080],
        available_tensor_instructions=None,
    )
    monkeypatch.setattr(arch_module, "CUDA", lambda target: arch)
    return arch


@pytest.mark.parametrize("op,selected", [("gemm_fp8", (20, 20)), ("grouped_gemm", (20, 0)), ("kda_chunk_o", (20, 0))])
def test_full_pool_template_dispatch_preserves_rejections(op, selected, offline_arch):
    configs = family_module(op, "spaces").get_configs()
    for workload, expected in zip(family_module(op, "cases").cases(holdout=True), selected):
        result = carver_rank(workload, Device("hopper", TARGETS["hopper"]), configs, 20)
        json.dumps(result, allow_nan=False)
        assert [r["config"] for r in result["configs"]] == configs
        assert len(result["ranking"]) == len(configs)
        assert result["selection"]["selected_count"] == expected
        assert result["selection"]["shortfall"] == 20 - expected
        for row in result["configs"]:
            assert (row["tile_cost"]["score"] is not None) == row["model"]["valid"]


def test_grouped_template_preserves_shapes_layouts_and_dtype(offline_arch):
    from tilelang.carver.template import GroupedMatmulTemplate

    template = GroupedMatmulTemplate(
        batch_sizes=(63, 77), N=128, K=96, trans_B=True, in_dtype="bfloat16", out_dtype="bfloat16", _arch=offline_arch
    )
    for m, func in zip((63, 77), template.equivalent_function()):
        buffers = [func.buffer_map[param] for param in func.params]
        assert [tuple(int(n) for n in b.shape) for b in buffers] == [(m, 96), (128, 96), (m, 128)]
        assert all(str(b.dtype) == "bfloat16" for b in buffers)


def test_kda_template_retains_scalar_producers_and_fuses_both_gemms(offline_arch):
    from tilelang.carver.template import KDAChunkTemplate
    from tilelang.carver.roller.policy import TensorCorePolicy
    from tvm import tirx as tir

    template = KDAChunkTemplate(batch=1, heads=2, sequence=96, dim=96, value_dim=80, chunk_size=48, _arch=offline_arch)
    func = template.equivalent_function()
    nodes = []
    tir.stmt_functor.post_order_visit(func.body, nodes.append)
    blocks = {n.name_hint: n for n in nodes if isinstance(n, tir.SBlock)}
    for name in ("ScaledQ", "GatedQ", "CausalA", "State", "Local", "Output"):
        assert name in blocks
    for name in ("ScaledQ", "GatedQ", "Output"):
        assert isinstance(blocks[name].body.value, tir.Cast)
        assert str(blocks[name].body.value.dtype) == "float16"
    assert any(isinstance(n, tir.Call) and getattr(n.op, "name", "") == "tirx.exp2" for n in nodes)
    policy = TensorCorePolicy.from_output_nodes(template.output_nodes, offline_arch)
    assert len(policy.ordered_nodes) == 3
    terminal = policy.ordered_nodes[-1]
    assert {edge.src_node for edge in terminal.inputs} == set(template.stage_nodes)


def test_kda_template_cpu_math_matches_independent_reference():
    import numpy as np
    from tilelang import tvm
    from tilelang.carver.template import KDAChunkTemplate

    # The C host backend is available in CUDA-only builds. FP32 checks graph
    # indexing and causal semantics; the structural test checks FP16 rounding.
    template = KDAChunkTemplate(batch=1, heads=2, sequence=8, dim=5, value_dim=7, chunk_size=4, in_dtype="float32", out_dtype="float32")
    func = template.equivalent_function()
    built = tvm.compile(func, target="c").jit(options=["-std=c++17"])
    rng = np.random.default_rng(13)
    inputs = [(rng.random(tuple(int(n) for n in func.buffer_map[p].shape)) - 0.5).astype("float32") for p in func.params[:-1]]
    q, gate, state, causal, values = inputs
    expected = ((q * 5**-0.5) * np.exp2(gate)) @ state + np.tril(causal) @ values
    output = tvm.runtime.empty(expected.shape, "float32", tvm.cpu())
    built(*[tvm.runtime.tensor(x) for x in inputs], output)
    np.testing.assert_allclose(output.numpy(), expected, atol=1e-6, rtol=1e-5)

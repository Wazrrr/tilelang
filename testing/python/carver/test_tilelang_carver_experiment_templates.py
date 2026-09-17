"""Carver templates preserve operation semantics and evaluate native pools."""

import pytest
import torch

from tilelang.carver.template import FlashAttentionTemplate, KDAChunkOutputTemplate


def test_attention_template_includes_normalization_mask_and_cast():
    template = FlashAttentionTemplate(seq_length=128, seq_kv_length=64, head_dim=64, is_causal=True)
    source = template.equivalent_function().script()
    assert template.params_as_dict()["is_causal"] is True
    assert "head_dim=64" in repr(template)
    assert all(name in source for name in ("Scores", "Scaled", "Maximum", "Exponentials", "Denominator", "Probabilities", "Numerator"))
    assert "if_then_else" in source and "exp" in source and 'Cast("float16"' in source
    with pytest.raises(ValueError, match="floating-point"):
        FlashAttentionTemplate(in_dtype="int8", accum_dtype="int32")


def test_kda_template_preserves_both_rounding_points_and_causal_term():
    template = KDAChunkOutputTemplate(batch_size=2, num_heads=4, seq_length=192, head_dim=96, value_dim=80, chunk_size=48)
    func = template.equivalent_function()
    source = func.script()
    assert tuple(int(n) for n in func.buffer_map[func.params[0]].shape) == (32, 48, 96)
    assert all(name in source for name in ("ScaledQ", "GatedQ", "Carried", "MaskedA", "Local"))
    assert "exp2" in source and "if_then_else" in source
    assert source.count('Cast("float16"') >= 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Carver queries actual CUDA capacities")
@pytest.mark.parametrize("name", ["attention_noncausal", "attention_causal", "kda_chunk_regular", "kda_chunk_tails"])
def test_graph_carver_ranks_every_final_case(name):
    from experiments.common.spec import Device, default_workloads, configurations
    from experiments.common.baselines import carver_rank
    from tilelang.tiletune import current_target

    w = next(w for w in default_workloads() if w.name == name)
    d = Device("test", current_target())
    configs = configurations(w, d)
    report = carver_rank(w, d, configs, 20)
    assert len(report["configs"]) == len(configs)
    assert report["selection"]["selected_count"] == 20
    for index in report["selection"]["selected_indices"]:
        record = report["configs"][index]
        model = record["model"]
        assert record["config"] == configs[index]
        assert record["tile_cost"]["score"] == (model["traffic_bytes"] + 1) * model["waves"] * model["key_tile_iterations"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Carver queries actual CUDA capacities")
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_matmul_uses_existing_template(dtype):
    from tilelang import tvm
    from tilelang.carver.template import MatmulTemplate
    from tilelang.carver.matmul_analysis import get_tensorized_func_and_tags

    # Cross-target template analysis only; no FP8 kernel runs on an A100.
    template = MatmulTemplate(M=512, N=512, K=512, in_dtype=dtype, out_dtype=dtype, accum_dtype="float32")
    _, tags = get_tensorized_func_and_tags(template.equivalent_function(), tvm.target.Target({"kind": "cuda", "arch": "sm_90a"}))
    assert tags["intrin_info"]["in_dtype"] == dtype
    assert tags["intrin_info"]["out_dtype"] == "float32"

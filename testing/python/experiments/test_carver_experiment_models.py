"""Carver templates preserve operation semantics and evaluate native pools."""

import pytest
import torch

from tilelang.carver.template import FlashAttentionTemplate, KDAChunkTemplate


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
    template = KDAChunkTemplate(batch_size=2, num_heads=4, sequence=192, key_dim=96, value_dim=80, chunk_size=48)
    func = template.equivalent_function()
    source = func.script()
    assert tuple(int(n) for n in func.buffer_map[func.params[0]].shape) == (32, 48, 96)
    assert all(name in source for name in ("ScaledQ", "GatedQ", "Carried", "MaskedA", "Local"))
    assert "exp2" in source and "if_then_else" in source
    assert source.count('Cast("float16"') >= 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Carver queries actual CUDA capacities")
@pytest.mark.parametrize("name", ["attention_noncausal", "attention_causal"])
def test_graph_carver_ranks_the_attention_final_cases(name):
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
        assert record["tile_cost"]["score"] == (model["traffic_bytes_per_cta"] + 1) * model["waves"]
        if w.op == "attention":
            p = w.parameters
            # All heads have contiguous 64/128-wide FP16 rows. Q and O are
            # visited once, while K/V include every tile the example loads.
            bm, bn = configs[index]["block_M"], configs[index]["block_N"]
            from math import ceil

            blocks = ceil(p["sequence"] / bm)
            kv_rows = sum(min(p["sequence"], ceil((q + 1) * bm / bn) * bn) if p.get("causal") else p["sequence"] for q in range(blocks))
            expected = (2 * p["sequence"] + 2 * kv_rows) * p["dim"] * 2 / blocks
            assert model["traffic_bytes_per_cta"] == pytest.approx(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Carver queries actual CUDA capacities")
def test_attention_traffic_preserves_once_only_io_and_clips_tail_tiles():
    from experiments.common.spec import Device, Workload
    from experiments.common.baselines import carver_rank
    from tilelang.tiletune import current_target

    config = dict(block_M=64, block_N=64, num_stages=0, threads=128)
    reports = []
    for sequence in (64, 128, 160):
        w = Workload("traffic", "attention", dict(batch=1, heads=4, sequence=sequence, dim=64, causal=False))
        report = carver_rank(w, Device("test", current_target()), [config], 1)
        reports.append(report["configs"][0]["model"])
    assert reports[0]["traffic_bytes_per_cta"] == 4 * 64 * 64 * 2
    assert reports[1]["traffic_bytes_per_cta"] == (2 * 64 + 2 * 128) * 64 * 2
    # Average over three query CTAs, including the 32-row tail; K/V are 160
    # rows per CTA, with no traffic charged for the padded final 32 rows.
    assert reports[2]["traffic_bytes_per_cta"] == pytest.approx((2 * 160 / 3 + 2 * 160) * 64 * 2)

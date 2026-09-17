"""Attention uses the original Carver model and preserves its rejected trials."""

from dataclasses import replace
import json

import pytest

from experiments.common.baselines import carver_rank, carver_support_reason
from experiments.common.spec import Device, TARGETS
from experiments.flash_attention.cases import cases
from experiments.flash_attention.spaces import get_configs


def test_attention_support_includes_both_semantics_and_preserves_backend_boundary():
    for workload in cases() + cases(holdout=True):
        for dtype in ("float16", "bfloat16"):
            assert carver_support_reason(replace(workload, dtype=dtype), Device("hopper", TARGETS["hopper"])) is None
        assert carver_support_reason(workload, Device("hip", dict(kind="hip", mcpu="gfx942")))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_attention_scores_and_feasibility_match_original_policy(dtype):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from experiments.flash_attention.carver import attention_template
    from experiments.gemm.carver import model_target
    from tilelang.carver.arch import CUDA
    from tilelang.carver.roller.policy import TensorCorePolicy
    from tilelang.tiletune.profiling.device_profile import current_target
    from tilelang import tvm

    device = Device("local", dict(tvm.target.Target(current_target()).export()))
    configs = get_configs()
    for workload in cases() + cases(holdout=True):
        workload = replace(workload, dtype=dtype)
        report = carver_rank(workload, device, configs, 20)
        json.dumps(report, allow_nan=False)
        assert [row["config"] for row in report["configs"]] == configs
        assert len(report["ranking"]) == 320
        arch = CUDA(model_target(device.target))
        template = attention_template(arch, **workload.parameters, dtype=dtype)
        policy = TensorCorePolicy.from_output_nodes(template.output_nodes, arch)
        qk, pv = policy.ordered_nodes
        expected = []
        for i, config in enumerate(configs):
            policy.pipeline_stage = max(1, config["num_stages"])
            steps = {qk: policy._assign_reduce_step(qk), pv: {axis.var.name: config["block_N"] for axis in pv.raxis}}
            td = policy.compute_tile_dict([1, config["block_M"], workload.parameters["dim"]], steps)
            valid = td.valid and policy.check_tile_shape_isvalid(td)
            valid = valid and all(policy._assign_block_size(n, td, config["threads"]) is not None for n in policy.ordered_nodes)
            row = report["configs"][i]
            assert row["model"]["shared_bytes"] == td.smem_cost
            assert row["model"]["valid"] == bool(valid)
            if valid:
                score = float((td.traffic + 1) * td.num_wave)
                assert row["tile_cost"]["score"] == score
                expected.append((score, i))
            else:
                assert row["status"] == "model_rejected" and row["tile_cost"]["score"] is None
        assert report["selection"]["selected_indices"] == [i for _, i in sorted(expected)[:20]]
        assert report["selection"]["shortfall"] == 20 - min(20, len(expected))


def test_empty_carver_selection_is_recorded_without_compilation(tmp_path):
    from experiments.common import execution
    from types import SimpleNamespace

    def no_compile(*args):
        raise AssertionError("empty Carver selection must not elaborate a kernel")

    configs = [get_configs()[0]]
    report = dict(
        metric="carver_traffic_waves",
        score_units="byte-waves",
        configs=[dict(index=0, config=configs[0], status="model_rejected", selected=False)],
        ranking=[dict(index=0, rank=1, tier="unknown", score=None)],
        selection=dict(requested_k=20, selected_indices=[], selected_count=0, shortfall=20, wall_time_ms=1),
    )
    result = execution.run_selected(
        SimpleNamespace(build=no_compile), configs, [13], TARGETS["hopper"], [], None, {}, tmp_path, report, report_name="carver"
    )
    assert result["status"] == "model_unavailable"
    assert "no fallback" in result["reason"]
    assert "winner" not in result
    records = json.loads((tmp_path / "outcomes.json").read_text())
    assert records[0]["original_index"] == 13 and records[0]["status"] == "model_rejected"
    assert json.loads((tmp_path / "compilation-census.json").read_text())["compiled_count"] == 0

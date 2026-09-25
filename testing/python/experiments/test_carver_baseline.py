"""The Carver adapter handles legacy targets and ranks the common grid."""

import pytest
import torch

from experiments.gemm.carver import model_target, rank_configs
from experiments.gemm.spaces import get_configs


def test_legacy_target_spelling_does_not_mutate_compile_target():
    target = {"kind": "cuda", "arch": "sm_90a"}
    assert str(model_target(target).attrs["arch"]) == "sm_90"
    assert target["arch"] == "sm_90a"
    assert str(model_target({"kind": "cuda", "arch": "sm_80"}).attrs["arch"]) == "sm_80"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_original_carver_ranks_common_grid(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tilelang.tiletune.profiling.device_profile import current_target

    configs = get_configs()
    result = rank_configs(configs, m=4096, n=4096, k=4096, dtype=dtype, target=current_target(), top_k=20)
    assert len(result["configs"]) == len(configs)
    assert result["selection"]["selected_count"] == 20
    for index in result["selection"]["selected_indices"]:
        record = result["configs"][index]
        assert record["model"]["valid"]
        assert record["tile_cost"]["score"] == (record["model"]["traffic_bytes"] + 1) * record["model"]["waves"]


@pytest.mark.parametrize("op", ["gemm", "attention", "gemm_fp8", "grouped_gemm"])
def test_every_experiment_family_has_a_carver_common_grid_adapter(op):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from experiments.common.baselines import carver_rank, carver_support_reason
    from experiments.common.spec import Device, configurations
    from experiments.families import family_module
    from tilelang.tiletune.profiling.device_profile import current_target

    device = Device("visible", current_target())
    expected_template = {
        "gemm": "MatmulTemplate",
        "attention": "FlashAttentionTemplate",
        "gemm_fp8": "FP8MatmulTemplate",
        "grouped_gemm": "GroupedMatmulTemplate",
    }[op]
    for workload in family_module(op, "cases").cases(holdout=True):
        all_configs = configurations(workload, device)
        configs = all_configs[:8]
        if op == "attention":
            # Select the second tile by value: pool expansion changes indices.
            configs += [c for c in all_configs if c["block_M"] == 64][:8]
        reason = carver_support_reason(workload, device)
        if op == "gemm_fp8" and device.target["arch"] == "sm_80":
            assert reason and "FP8" in reason
            continue
        assert reason is None
        result = carver_rank(workload, device, configs, top_k=2)
        assert result["template"] == expected_template
        assert len(result["configs"]) == len(configs)
        assert [record["config"] for record in result["configs"]] == configs
        assert result["selection"]["selected_count"] == 2
        assert all(result["configs"][index]["model"]["valid"] for index in result["selection"]["selected_indices"])


def test_kda_intra_operation_has_a_carver_adapter():
    from experiments.common.baselines import carver_rank, carver_support_reason
    from experiments.common.spec import Device, TARGETS
    from experiments.families import family_module

    workload = family_module("kda_chunk_intra_token_parallel", "cases").cases(holdout=True)[0]
    configs = family_module("kda_chunk_intra_token_parallel", "spaces").get_configs()
    device = Device("ampere", TARGETS["ampere"])
    assert carver_support_reason(workload, device) is None
    report = carver_rank(workload, device, configs, 8)
    assert report["model"] == "carver_kda_intra_traffic_waves"
    assert report["template"] == "kda_intra_token_parallel"
    assert len(report["ranking"]) == len(configs)

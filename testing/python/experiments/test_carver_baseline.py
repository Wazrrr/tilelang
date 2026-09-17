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


@pytest.mark.parametrize("op", ["gemm", "attention", "kda_chunk_o", "gemm_fp8", "grouped_gemm"])
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
        "kda_chunk_o": "KDAChunkTemplate",
        "gemm_fp8": "FP8MatmulTemplate",
        "grouped_gemm": "GroupedMatmulTemplate",
    }[op]
    for workload in family_module(op, "cases").cases(holdout=True):
        all_configs = configurations(workload, device)
        configs = all_configs[:8] + all_configs[64:72] if op == "attention" else all_configs[:8]
        if op == "grouped_gemm":
            assert "no block-scaled 2-CTA grouped-MXFP8 model" in carver_support_reason(workload, device)
            with pytest.raises(ValueError, match="no block-scaled"):
                carver_rank(workload, device, configs, top_k=2)
            continue
        assert carver_support_reason(workload, device) is None
        result = carver_rank(workload, device, configs, top_k=2)
        assert result["template"] == expected_template
        assert len(result["configs"]) == len(configs)
        assert [record["config"] for record in result["configs"]] == configs
        assert result["selection"]["selected_count"] == 2
        assert all(result["configs"][index]["model"]["valid"] for index in result["selection"]["selected_indices"])
        if op == "attention":
            assert all(not record["model"]["valid"] for record in result["configs"][:8])

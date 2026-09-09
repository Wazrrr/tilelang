"""The Carver adapter handles legacy targets and ranks the common grid."""

import pytest
import torch

from experiments.gemm.carver import model_target, rank_configs
from experiments.gemm.kernel import get_configs


def test_legacy_target_spelling_does_not_mutate_compile_target():
    target = {"kind": "cuda", "arch": "sm_90a"}
    assert str(model_target(target).attrs["arch"]) == "sm_90"
    assert target["arch"] == "sm_90a"
    assert str(model_target({"kind": "cuda", "arch": "sm_80"}).attrs["arch"]) == "sm_80"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_original_carver_ranks_common_grid(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tilelang.tiletune.device_profile import current_target

    configs = get_configs()
    result = rank_configs(configs, m=4096, n=4096, k=4096, dtype=dtype, target=current_target(), top_k=20)
    assert len(result["configs"]) == len(configs)
    assert result["selection"]["selected_count"] == 20
    for index in result["selection"]["selected_indices"]:
        record = result["configs"][index]
        assert record["model"]["valid"]
        assert record["tile_cost"]["score"] == (record["model"]["traffic_bytes"] + 1) * record["model"]["waves"]

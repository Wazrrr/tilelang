"""A small grid shares chip memory bandwidth, while compute remains per SM."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tiletune_core.ranking import apply_ranking_metric


def score(grid, *, bytes_per_cta=1024, flops_per_cta=0, resident=1):
    rates = dict(global_bytes_per_cycle=1, copy_latency_cycles=0, barrier_cycles=0, gemm_flops_per_cycle=1)
    pipeline = dict(
        performance_model=rates,
        unknown=[],
        iterations={"max": 1},
        effective_buffer_depth=1,
        phases=[
            dict(
                operation=0,
                inside_loop=True,
                work=dict(gemm_flops=flops_per_cta, shared_bytes=0, elementwise_ops=0, exp_ops=0, reduction_ops=0),
            )
        ],
        input_bytes_per_iteration=bytes_per_cta,
        producer_copies_per_iteration=0,
        overlap_eligible=False,
        outside_loop_bytes=0,
        outside_loop_input_copies=0,
        cta_work=dict(precision="exact", groups=[dict(iterations=1, count=grid)], repetitions=1, grid_blocks=grid),
    )
    before = deepcopy(pipeline)
    waves = dict(
        unknown=[],
        resident_blocks_per_sm_estimate=resident,
        device_limits={"sm_count": 32},
        grid_blocks=grid,
        num_waves_estimate=(grid + 32 * resident - 1) // (32 * resident),
    )
    result = apply_ranking_metric(
        {"unknown": []},
        waves,
        pipeline,
        SimpleNamespace(ranking_metric="pipeline_time", performance_model=rates),
        SimpleNamespace(matched=True),
    )
    assert pipeline == before
    return result["score"]


def test_memory_service_conserves_chip_bandwidth_for_small_grids():
    assert score(1) == pytest.approx(32)
    assert score(16) == pytest.approx(512)
    assert score(32) == pytest.approx(1024)
    assert score(64) == pytest.approx(2048)


@pytest.mark.parametrize("resident", [1, 2, 3, 8])
@pytest.mark.parametrize("grid", [1, 16, 32, 33, 64, 80, 96, 97, 105, 128, 200])
def test_chip_memory_service_counts_real_ctas_for_any_wave_fill(grid, resident):
    assert score(grid, resident=resident) == pytest.approx(grid * 1024 / 32)


def test_idle_sms_cannot_supply_compute_to_another_sm():
    assert score(1, bytes_per_cta=0, flops_per_cta=1024) == 1024
    assert score(32, bytes_per_cta=0, flops_per_cta=1024) == 1024
    assert score(64, bytes_per_cta=0, flops_per_cta=1024) == 2048


def test_partial_last_wave_does_not_contend_with_absent_ctas():
    # Four CTAs per SM: three in the first wave, one in the last.
    assert score(128, resident=3) == pytest.approx(4096)
    assert score(128, resident=3, bytes_per_cta=0, flops_per_cta=1024) == 4096
    # The tail's nine CTAs share chip bandwidth, but each keeps its SM compute.
    assert score(105, resident=3) == pytest.approx(3360)
    assert score(105, resident=3, bytes_per_cta=0, flops_per_cta=1024) == 4096

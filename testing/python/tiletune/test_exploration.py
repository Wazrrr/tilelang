import pytest
from tilelang.tiletune import TileTuneConfig
from tilelang.tiletune.ranking import rank_records, select_with_exploration
from tilelang.tiletune.register_pressure import analyze_register_policy


def records(scored=30, unknown=10):
    return [
        dict(
            index=i,
            config=dict(implementation="a" if i % 2 else "b", block=i),
            status="analyzed",
            tile_cost=dict(score=i if i < scored else None),
            pre_lowering=dict(would_reject=False),
        )
        for i in range(scored + unknown)
    ]


@pytest.mark.parametrize("scored,unknown,expected", [(30, 10, 4), (5, 20, 15), (30, 2, 2), (0, 30, 20), (30, 0, 0)])
def test_exact_attempts_and_fill(scored, unknown, expected):
    rows = records(scored, unknown)
    ranking = rank_records(rows)
    selected, explored = select_with_exploration(ranking, rows, 20)
    assert len(selected) == min(20, scored + unknown)
    assert len(set(selected)) == len(selected)
    assert len(explored) == expected
    assert (selected, explored) == select_with_exploration(ranking, list(reversed(rows)), 20)
    # Results are frozen; failures cannot enlarge the selected attempt list.
    for i in selected:
        rows[i]["status"] = "compilation_failed"
    assert (selected, explored) == select_with_exploration(ranking, rows, 20)


def test_explicit_policy_and_resource_rejections_are_never_explored():
    rows = records(0, 30)
    rows[0]["pre_lowering"] = dict(would_reject=True, keep=True, classification="policy_rejection")
    rows[1]["pre_lowering"] = dict(would_reject=True, keep=True, classification="resource_violation")
    selected, _ = select_with_exploration(rank_records(rows), rows, 20)
    assert 0 not in selected and 1 not in selected


def test_exploration_reservation_does_not_split_scored_ties():
    rows = records(4, 2)
    for row in rows[:4]:
        row["tile_cost"]["score"] = 10
    selected, explored = select_with_exploration(rank_records(rows), rows, 3, fraction=1 / 3)
    assert selected[:4] == [0, 1, 2, 3]
    assert len(explored) == 1 and len(selected) == 5


def test_logical_demand_is_uncertainty_when_spilling_is_permitted():
    pressure = dict(
        budget=255,
        hardware_register_cap=255,
        modeled_lower_bound=256,
        tile_liveness=dict(peak_registers_per_block_estimate=32768, computing_threads_estimate=128),
    )
    permitted = analyze_register_policy(pressure, TileTuneConfig(max_spill_bytes=None, max_local_bytes=None))
    assert permitted["decision"]["classification"] == "allocation_uncertainty"
    assert not permitted["decision"]["would_reject"]
    for kwargs in ({}, dict(register_cap=128, max_spill_bytes=None, max_local_bytes=None)):
        strict = analyze_register_policy(pressure, TileTuneConfig(**kwargs))
        assert strict["decision"]["classification"] == "policy_rejection"
        assert strict["decision"]["would_reject"]


def test_proven_launch_limit_is_not_allocation_uncertainty():
    pressure = dict(
        budget=255, hardware_register_cap=255, tile_liveness=dict(peak_registers_per_block_estimate=2048, computing_threads_estimate=2048)
    )
    result = analyze_register_policy(pressure, TileTuneConfig(max_spill_bytes=None, max_local_bytes=None), dict(max_threads_per_block=1024))
    assert result["decision"]["would_reject"]
    assert result["decision"]["classification"] == "resource_violation"

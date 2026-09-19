"""Saved memory facts are resolved independently of oracle labels and scores."""

from copy import deepcopy

import pytest

from experiments.replay_memory import memory_inputs
from tiletune_core.memory import score_memory


def record():
    region = dict(buffer="input", scope="global", ranges=[dict(min="k * 96", extent="96")])
    return dict(
        pressure=dict(logical_storage=[dict(buffer="input", scope="global", logical_bits=4096 * 16, logical_elements=4096)]),
        tile_propagation=dict(operations=[dict(index=7, unknown=False, reads=[region], writes=[], loops=[dict(kind="0", extent="43")])]),
        modules=dict(memory_traffic=dict(input_tiles=[]), waves=dict(grid_blocks=133)),
    )


def test_replay_uses_requested_tile_and_ignores_all_measured_information():
    r = record()
    inputs, unknown = memory_inputs(r, 132)
    assert not unknown
    assert inputs["accesses"][0]["bytes"] == 96 * 2
    assert score_memory(**inputs)["logical_byte_waves"] == 96 * 2 * 43 * 2
    changed = deepcopy(r)
    changed.update(config=dict(block_K=1), latency_ms=0.0001, winner=True, tile_cost=dict(score=1))
    changed["modules"]["pipeline_overlap"] = dict(timing=dict(cycles=1))
    assert memory_inputs(changed, 132) == (inputs, unknown)


def test_symbolic_visits_require_a_saved_operation_bound():
    r = record()
    r["tile_propagation"]["operations"][0]["loops"][0]["extent"] = "min(43, bx + 1)"
    with pytest.raises(ValueError, match="no unambiguous archived loop-visit bound"):
        memory_inputs(r, 132)
    r["modules"]["memory_traffic"]["input_tiles"] = [dict(operation=7, visits_per_block=43)]
    inputs, unknown = memory_inputs(r, 132)
    assert not unknown and inputs["accesses"][0]["visits"] == 43
    r["tile_propagation"]["operations"][0]["unknown"] = True
    assert memory_inputs(r, 132)[1] == ["operation 7: unresolved memory effects"]

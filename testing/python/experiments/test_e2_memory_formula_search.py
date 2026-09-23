from experiments.e2_memory_formula_search import (
    adjusted_byte_waves,
    encode_bytes_accesses_depth,
    encode_bytes_depth_accesses,
    explicit_rank_key,
    grid_rank_key,
    grid_only_adjusted_byte_waves,
)


def test_b200_order_moves_depth_before_access_count():
    # Previous H200 prefers fewer requests before depth; B200 does the reverse.
    shallow_few = (100, 1, 1)
    deep_many = (100, 8, 100)
    assert encode_bytes_accesses_depth(*shallow_few) < encode_bytes_accesses_depth(*deep_many)
    assert encode_bytes_depth_accesses(*deep_many) < encode_bytes_depth_accesses(*shallow_few)


def test_underfill_correction_is_integer_and_access_damped():
    common = dict(byte_waves=100, sm_count=132, target_waves=3)
    assert adjusted_byte_waves(**common, grid_blocks=396, accesses_per_cta=1) == 100
    assert adjusted_byte_waves(**common, grid_blocks=132, accesses_per_cta=0) == 300
    assert adjusted_byte_waves(**common, grid_blocks=132, accesses_per_cta=132) == 200


def test_grid_only_underfill_correction_matches_three_s_over_g():
    common = dict(byte_waves=100, sm_count=132, target_waves=3)
    assert grid_only_adjusted_byte_waves(**common, grid_blocks=396) == 100
    assert grid_only_adjusted_byte_waves(**common, grid_blocks=132) == 300
    assert grid_only_adjusted_byte_waves(**common, grid_blocks=264) == 150


def test_explicit_rank_key_uses_requested_lexicographic_order():
    baseline = explicit_rank_key(100, 128, 16, 4, 32)
    assert explicit_rank_key(99, 1, 1, 1, 1000) < baseline
    assert explicit_rank_key(100, 256, 1, 1, 1000) < baseline
    assert explicit_rank_key(100, 128, 32, 1, 1000) < baseline
    assert explicit_rank_key(100, 128, 16, 5, 1000) < baseline
    assert explicit_rank_key(100, 128, 16, 4, 16) < baseline


def test_grid_rank_key_does_not_double_count_per_cta_accesses():
    baseline = grid_rank_key(100, 128, 4, 32)
    assert grid_rank_key(99, 1, 1, 1000) < baseline
    assert grid_rank_key(100, 256, 1, 1000) < baseline
    assert grid_rank_key(100, 128, 5, 1000) < baseline
    assert grid_rank_key(100, 128, 4, 16) < baseline


def test_byte_band_dominates_depth_and_access_extremes():
    previous = encode_bytes_depth_accesses(100, 1, 100)
    following = encode_bytes_depth_accesses(101, 65535, 0)
    assert previous < following

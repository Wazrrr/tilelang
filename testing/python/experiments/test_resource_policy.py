"""The strict B200 E3 limits must retain every observed exhaustive oracle."""

from types import SimpleNamespace

import pytest

from experiments.common.resource_policy import b200_post_compile_policy
from tilelang.contrib.cuda_resource_info import parse_ptxas_output
from tilelang.tiletune import check_compiler_resources


OBSERVED_E1_ORACLES = [
    ("gemm_decode", "gemm", 0, 0, 0),
    ("gemm_prefill", "gemm", 0, 0, 0),
    ("gemm_ffn_down", "gemm", 0, 0, 0),
    ("gemm_square", "gemm", 0, 0, 0),
    ("gemm_square_large", "gemm", 0, 0, 0),
    ("attention_short_causal", "attention", 0, 0, 0),
    ("attention_batched_causal", "attention", 0, 0, 0),
    ("attention_noncausal", "attention", 0, 0, 0),
    ("attention_causal", "attention", 0, 0, 0),
    ("attention_long_causal", "attention", 0, 0, 0),
    ("kda_intra_short", "kda_chunk_intra_token_parallel", 0, 0, 0),
    ("kda_intra_medium", "kda_chunk_intra_token_parallel", 0, 0, 0),
    ("kda_intra_regular", "kda_chunk_intra_token_parallel", 0, 0, 0),
    ("kda_intra_batched", "kda_chunk_intra_token_parallel", 0, 0, 0),
    ("kda_intra_long", "kda_chunk_intra_token_parallel", 0, 0, 0),
    ("gemm_fp8_decode", "gemm_fp8", 0, 0, 0),
    ("gemm_fp8_prefill", "gemm_fp8", 0, 0, 0),
    ("gemm_fp8_ffn_down", "gemm_fp8", 0, 0, 0),
    ("gemm_fp8_square", "gemm_fp8", 12, 8, 8),
    ("gemm_fp8_square_large", "gemm_fp8", 12, 8, 8),
    ("grouped_gemm_decode", "grouped_gemm", 0, 0, 0),
    ("grouped_gemm_prefill", "grouped_gemm", 0, 0, 0),
    ("grouped_gemm_aligned", "grouped_gemm", 0, 0, 0),
    ("grouped_gemm_down_aligned", "grouped_gemm", 0, 0, 0),
    ("grouped_gemm_ragged", "grouped_gemm", 0, 0, 0),
]


@pytest.mark.parametrize("name,op,stores,loads,local", OBSERVED_E1_ORACLES)
def test_each_observed_b200_oracle_fits_its_family_policy(name, op, stores, loads, local):
    target = dict(kind="cuda", arch="sm_100a")
    policy = b200_post_compile_policy(SimpleNamespace(name=name, op=op), target)
    report = parse_ptxas_output(
        f"""ptxas info : Compiling entry function 'kernel' for 'sm_100'
ptxas info : Function properties for kernel
    {local} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads
ptxas info : Used 255 registers
"""
    )
    assert check_compiler_resources(report, ["kernel"], policy, target=target)["keep"]


@pytest.mark.parametrize(
    "op,limit",
    [
        ("gemm", 0),
        ("attention", 0),
        ("kda_chunk_intra_token_parallel", 0),
        ("gemm_fp8", 16),
        ("grouped_gemm", 0),
    ],
)
def test_b200_family_policy_is_a_real_rejection_boundary(op, limit):
    target = dict(kind="cuda", arch="sm_100a")
    policy = b200_post_compile_policy(SimpleNamespace(op=op), target)
    report = parse_ptxas_output(
        f"""ptxas info : Compiling entry function 'kernel' for 'sm_100'
ptxas info : Function properties for kernel
    0 bytes stack frame, {limit + 1} bytes spill stores, 0 bytes spill loads
ptxas info : Used 168 registers
"""
    )
    assert not check_compiler_resources(report, ["kernel"], policy, target=target)["keep"]


@pytest.mark.parametrize(
    "target",
    [dict(kind="cuda", arch="sm_90a"), dict(kind="cuda", arch="sm_120"), dict(kind="hip", mcpu="gfx950")],
)
def test_b200_limits_are_not_applied_to_other_backends(target):
    assert b200_post_compile_policy(SimpleNamespace(op="attention"), target) is None

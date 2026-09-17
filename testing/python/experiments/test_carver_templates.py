"""Canonical Carver template selection for every experiment family."""

from types import SimpleNamespace

import pytest

from experiments.common.carver import workload_template
from experiments.gemm.carver import model_target


def _workload(name, op, parameters, dtype="float16"):
    return SimpleNamespace(name=name, op=op, parameters=parameters, dtype=dtype)


def _offline_hopper():
    return SimpleNamespace(target=model_target({"kind": "cuda", "arch": "sm_90a"}))


@pytest.mark.parametrize(
    ("workload", "expected"),
    [
        (_workload("gemm", "gemm", {"m": 128, "n": 128, "k": 128, "transpose_b": True}), "MatmulTemplate"),
        (
            _workload(
                "fp8",
                "gemm_fp8",
                {"m": 128, "n": 128, "k": 128, "transpose_b": True},
                dtype="float8_e4m3fn",
            ),
            "FP8MatmulTemplate",
        ),
        (
            _workload("attention", "attention", {"batch": 1, "heads": 2, "sequence": 128, "dim": 64}),
            "FlashAttentionTemplate",
        ),
        (
            _workload(
                "grouped",
                "grouped_gemm",
                {"batch_sizes": [31, 65], "n": 128, "k": 64, "transpose_b": True},
            ),
            "GroupedMatmulTemplate",
        ),
        (
            _workload(
                "kda",
                "kda_chunk_o",
                {"batch": 1, "heads": 2, "sequence": 128, "dim": 64, "value_dim": 64, "chunk_size": 64},
            ),
            "KDAChunkTemplate",
        ),
    ],
)
def test_each_experiment_family_selects_its_canonical_template(workload, expected):
    configs = [{"block_M": 64}] if workload.op == "grouped_gemm" else None
    template = workload_template(workload, configs, arch=_offline_hopper())
    assert type(template).__name__ == expected


@pytest.mark.parametrize(
    ("kernel_dtype", "model_dtype"),
    [("float8_e4m3fn", "float8_e4m3"), ("float8_e5m2", "float8_e5m2")],
)
def test_fp8_template_preserves_kernel_dtype_and_uses_tensorizable_model_dtype(kernel_dtype, model_dtype):
    from tilelang.carver.matmul_analysis import get_tensorized_func_and_tags

    workload = _workload(
        "fp8",
        "gemm_fp8",
        {"m": 128, "n": 128, "k": 128, "transpose_b": True},
        dtype=kernel_dtype,
    )
    template = workload_template(workload, arch=_offline_hopper())
    assert template.kernel_dtype == kernel_dtype
    assert template.in_dtype == model_dtype
    assert template.out_dtype == "bfloat16"
    _, tags = get_tensorized_func_and_tags(
        template.equivalent_function(), template.arch.target, allow_gemv=True
    )
    assert tags


def test_fused_templates_model_the_full_semantic_graph():
    attention = workload_template(
        _workload(
            "attention",
            "attention",
            {"batch": 1, "heads": 2, "sequence": 128, "dim": 64, "causal": True},
        ),
        arch=_offline_hopper(),
    )
    attention_ir = attention.equivalent_function().script()
    for stage in ("Scores", "Scaled", "Maximum", "Exponentials", "Denominator", "Probabilities", "Numerator"):
        assert stage in attention_ir

    kda = workload_template(
        _workload(
            "kda",
            "kda_chunk_o",
            {"batch": 1, "heads": 2, "sequence": 128, "dim": 64, "value_dim": 64, "chunk_size": 64},
        ),
        arch=_offline_hopper(),
    )
    kda_ir = kda.equivalent_function().script()
    for stage in ("ScaledQ", "GatedQ", "Carried", "MaskedA", "Local"):
        assert stage in kda_ir


def test_grouped_template_preserves_padded_cta_domain():
    workload = _workload(
        "grouped",
        "grouped_gemm",
        {"batch_sizes": [31, 65], "n": 128, "k": 64, "transpose_b": True},
    )
    template = workload_template(workload, [{"block_M": 64}], arch=_offline_hopper())
    assert template.M == 3 * 64
    assert template.block_m == 64
    assert template.in_dtype == template.out_dtype == "float16"
    assert template.accum_dtype == "float32"

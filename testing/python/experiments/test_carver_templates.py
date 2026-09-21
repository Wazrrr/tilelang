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
def test_fp8_template_preserves_kernel_dtype_and_explicit_scales(kernel_dtype, model_dtype):
    from tilelang.carver.template import FP8MatmulTemplate

    # This standalone legacy template is no longer the FP8 experiment adapter.
    template = FP8MatmulTemplate(
        M=128,
        N=128,
        K=128,
        trans_B=True,
        kernel_dtype=kernel_dtype,
        compute_dtype=model_dtype,
        _arch=_offline_hopper(),
    )
    assert template.kernel_dtype == kernel_dtype
    assert template.in_dtype == model_dtype
    assert template.out_dtype == "bfloat16"
    func = template.equivalent_function()
    assert len(func.params) == 5
    assert {str(b.name) for b in func.buffer_map.values()} >= {"ScaleA", "ScaleB"}
    assert "Partial" in func.script() and "Scaled" in func.script()


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


@pytest.mark.parametrize("op", ["gemm_fp8", "kda_chunk_intra_token_parallel"])
def test_deferred_experiment_templates_are_explicitly_unsupported(op):
    with pytest.raises((NotImplementedError, ValueError), match="Carver"):
        workload_template(_workload("deferred", op, {}), arch=_offline_hopper())


def test_grouped_template_preserves_padded_cta_domain():
    workload = _workload(
        "grouped",
        "grouped_gemm",
        {"batch_sizes": [31, 65], "n": 128, "k": 64, "transpose_b": True},
    )
    template = workload_template(workload, [{"block_M": 64}], arch=_offline_hopper())
    assert template.M == 3 * 64
    assert template.block_m == 64

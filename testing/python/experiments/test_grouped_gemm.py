"""Grouped GEMM registration, fixed metadata, and direct example reuse."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.common.spec import Device, TARGETS, Workload, configuration_space, default_workloads, load_manifest
from experiments.grouped_gemm.cases import cases, training_cases
from experiments.grouped_gemm.spaces import get_configs
from experiments.suite import core_cases, study_plan

ROOT = Path(__file__).resolve().parents[3]
CONFIG = dict(block_M=64, block_N=64, block_K=64, num_stages=2, threads=128)


@pytest.mark.parametrize(
    "module,argv",
    [
        ("system.run", ["--plan"]),
        ("census", ["--suite", "final", "--plan"]),
        ("tiletune.run", ["--suite", "full", "--device", "hopper", "--plan"]),
    ],
)
def test_planning_uses_only_standard_library_and_creates_no_results(module, argv, tmp_path):
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from experiments.grouped_gemm.{module} import main
assert main({argv!r}, **({{'family': 'grouped_gemm'}} if {module!r} == 'system.run' else {{}})) == 0
assert not any(name.split('.')[0] in ('tilelang', 'torch', 'tvm', 'xgboost') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)
    assert not list(tmp_path.iterdir())


def test_default_matrix_includes_grouped_and_has_frozen_disjoint_splits():
    assert default_workloads() == core_cases("final")
    assert len(default_workloads()) == 10
    assert sum(w.op == "grouped_gemm" for w in default_workloads()) == 2
    _, frozen = load_manifest(json.loads((ROOT / "experiments/manifests/grouped_gemm_final.json").read_text()))
    assert frozen == cases(holdout=True) == core_cases("final", ["grouped_gemm"])
    all_cases = training_cases() + cases() + cases(holdout=True)
    from experiments.xgboost.data import canonical_workload, digest

    assert len({digest(canonical_workload(w)) for w in all_cases}) == len(all_cases)
    plan = study_plan("full", [Device("hopper", TARGETS["hopper"])], families=["grouped_gemm"])
    assert plan["families"] == ["grouped_gemm"]
    assert [len(plan["splits"][key]) for key in ("train", "validation", "test")] == [2, 1, 2]
    assert all(s["indices"] == list(range(192)) for s in plan["subsets"]["hopper"].values())
    assert len(core_cases("full", ["gemm", "grouped_gemm"])) == 4


def test_pool_identity_subsets_and_system_variants():
    from experiments.common.system import system_plan, VARIANTS

    w = cases()[0]
    spaces = [configuration_space(w, Device(target, TARGETS[target])) for target in ("ampere", "hopper", "mi355x")]
    assert spaces[0] == spaces[1] == spaces[2]
    space = spaces[0]
    assert space["configs"] == get_configs()
    assert len(set(space["config_ids"])) == space["candidate_count"] == 192
    assert space["rejected_count"] == space["alias_count"] == space["budget_omitted_count"] == 0
    assert space["configs"][60] == CONFIG
    assert space["configs"][125] == dict(CONFIG, block_N=128, threads=256)
    device = Device("hopper", TARGETS["hopper"])
    subset = [space["configs"][i] for i in (125, 60)]
    assert configuration_space(replace(w, configs=subset), device)["configs"] == subset
    with pytest.raises(ValueError, match="subset"):
        configuration_space(replace(w, configs=[dict(CONFIG, block_M=32)]), device)
    plan = system_plan("grouped_gemm", indices=[125, 60])
    assert len(plan) == 10
    assert {row["variant"] for row in plan} == set(VARIANTS)
    assert all(row["indices"] == [125, 60] for row in plan)


@pytest.mark.parametrize("sizes", [[], [0, 64], [-1], [True], [1.5], "64,128", (64, 128)])
def test_invalid_group_sizes_rejected(sizes):
    with pytest.raises(ValueError, match="batch_sizes"):
        Workload("bad", "grouped_gemm", dict(batch_sizes=sizes, n=128, k=128))


def test_dtype_layout_and_carver_support():
    from experiments.common.baselines import carver_support_reason
    from experiments.xgboost.data import canonical_workload

    w = cases()[0]
    with pytest.raises(ValueError, match="float16 and bfloat16"):
        replace(w, dtype="float32")
    with pytest.raises(ValueError, match="transpose_b must be a bool"):
        replace(w, parameters=dict(w.parameters, transpose_b=1))
    implicit = replace(w, parameters={k: v for k, v in w.parameters.items() if k != "transpose_b"})
    assert canonical_workload(implicit) == canonical_workload(w)
    assert carver_support_reason(w, Device("hopper", TARGETS["hopper"])) is None


def test_example_and_adapter_sources_invalidate_baselines_and_models():
    from experiments.utils.baseline_store import measurement_sources
    from experiments.utils.cli import source_hashes
    from experiments.xgboost.data import domain, make_context

    sources = source_hashes("experiments/common/kernels.py")
    baseline = measurement_sources(["grouped_gemm"])
    args = (cases()[0], "portable.grouped_gemm", TARGETS["hopper"], "H200", "event")
    before = make_context(*args, sources)
    for path in (
        "examples/grouped_gemm/example_grouped_gemm_fwd.py",
        "experiments/grouped_gemm/kernel.py",
        "experiments/grouped_gemm/reference.py",
        "experiments/grouped_gemm/spaces.py",
    ):
        assert before["kernel_sha256"][path] == baseline[path] == sources[path]
        assert domain(make_context(*args, dict(sources, **{path: "changed"}))) != domain(before)


@pytest.mark.parametrize("transpose_b", [False, True])
def test_inputs_reference_and_fixed_offsets(transpose_b):
    import torch
    from experiments.grouped_gemm.kernel import make_case

    w = Workload("boundary", "grouped_gemm", dict(batch_sizes=[1, 63, 65], n=33, k=17, transpose_b=transpose_b))
    case = make_case(w)
    inputs = case.inputs("cpu", torch.Generator().manual_seed(123))
    repeated = case.inputs("cpu", torch.Generator().manual_seed(123))
    assert all(torch.equal(a, b) for a, b in zip(inputs, repeated))
    a, b, sizes, offsets, padded = inputs
    assert sizes.tolist() == [1, 63, 65]
    assert offsets.tolist() == [0, 1, 64]
    assert padded.tolist() == [0, 64, 128]
    expected = torch.empty((129, 33), dtype=a.dtype)
    for group, start, end in ((0, 0, 1), (1, 1, 64), (2, 64, 129)):
        weight = b[group].T if transpose_b else b[group]
        expected[start:end] = a[start:end].double() @ weight.double()
    case.check([case.reference(*inputs)], [expected])
    with pytest.raises(AssertionError):
        case.check([torch.zeros_like(expected)], [expected])
    with pytest.raises(ValueError, match="block_M=64"):
        case.build(**dict(CONFIG, block_M=32))


@pytest.mark.parametrize("w", cases(holdout=True), ids=lambda w: w.name)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_adapter_program_equals_example(w, dtype):
    from tilelang import tvm
    from examples.grouped_gemm.example_grouped_gemm_fwd import grouped_gemm
    from experiments.common.kernels import make_case

    w = replace(w, dtype=dtype)
    case = make_case(w)
    assert all(isinstance(cell.cell_contents, (int, float, str, bool, type(None))) for cell in case.build.__closure__ or [])
    p = w.parameters
    example = grouped_gemm.get_tir(
        K=p["k"], N=p["n"], batch_sizes_list=tuple(p["batch_sizes"]), trans_b=p["transpose_b"], dtype=dtype, **CONFIG
    )
    assert case.out_idx is None
    tvm.ir.assert_structural_equal(case.build(**CONFIG), example)


@pytest.mark.parametrize("sizes,n,k", [([64, 128], 128, 128), ([1, 63, 65], 97, 80)])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_grouped_gemm_on_gpu(sizes, n, k, transpose_b, dtype):
    import torch
    import tilelang
    from experiments.common.kernels import make_case
    from tilelang.tiletune import current_target

    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm required")
    torch.backends.cuda.matmul.allow_tf32 = False
    w = Workload("gpu", "grouped_gemm", dict(batch_sizes=sizes, n=n, k=k, transpose_b=transpose_b), dtype=dtype)
    case = make_case(w)
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    expected = case.reference(*inputs)
    for config in (CONFIG, dict(CONFIG, block_N=128, threads=256)):
        kernel = tilelang.compile(
            case.build(**config),
            target=current_target(),
            execution_backend="tvm_ffi",
            out_idx=case.out_idx,
            pass_configs=case.pass_configs,
        )
        case.check([kernel(*inputs)], [expected])


@pytest.mark.parametrize("transpose_b", [False, True])
def test_grouped_compilation_preserves_output_contract_on_gpu(transpose_b):
    import torch
    import tilelang
    from tilelang.autotuner.grouped_compile import compile_grouped_unit_tvm_ffi
    from tilelang.autotuner.param import CompileArgs
    from tilelang.tiletune import current_target
    from experiments.common.kernels import make_case

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    w = Workload("grouped", "grouped_gemm", dict(batch_sizes=[1, 63, 65], n=97, k=80, transpose_b=transpose_b))
    case = make_case(w)
    configs = [CONFIG, dict(CONFIG, block_N=128, threads=256)]
    before = [case.build(**c).script() for c in configs]
    results = compile_grouped_unit_tvm_ffi(
        list(enumerate(configs)),
        CompileArgs(target=tilelang.tvm.target.Target(current_target()), out_idx=case.out_idx, execution_backend="tvm_ffi"),
        case.build,
    )
    inputs = case.inputs("cuda", torch.Generator(device="cuda").manual_seed(123))
    expected = case.reference(*inputs)
    assert len(results) == len(configs)
    for _, _, kernel, error in results:
        assert error is None, error
        case.check([kernel(*inputs)], [expected])
    assert [case.build(**c).script() for c in configs] == before

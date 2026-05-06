from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
import types


def _load_selector_module():
    try:
        return importlib.import_module("tilelang.carver.config_selector.hopper_hybrid_v1")
    except Exception:
        pass

    module_name = "_selector_test_pkg.config_selector.hopper_hybrid_v1"
    if module_name in sys.modules:
        return sys.modules[module_name]

    for pkg_name in ("_selector_test_pkg", "_selector_test_pkg.roller", "_selector_test_pkg.config_selector"):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            sys.modules[pkg_name] = pkg

    hint_mod = types.ModuleType("_selector_test_pkg.roller.hint")
    hint_mod.Hint = type("Hint", (), {})
    sys.modules["_selector_test_pkg.roller.hint"] = hint_mod

    raster_mod = types.ModuleType("_selector_test_pkg.roller.rasterization")
    raster_mod.NoRasterization = type("NoRasterization", (), {})
    sys.modules["_selector_test_pkg.roller.rasterization"] = raster_mod

    module_path = Path(__file__).resolve().parents[3] / "tilelang/carver/config_selector/hopper_hybrid_v1.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SELECTOR = _load_selector_module()


@dataclass
class _FakeArch:
    platform: str = "CUDA"
    sm_version: int = 90
    smem_cap: int = 228 * 1024


def _candidate_grid() -> list[dict]:
    return [
        {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 0, "thread_num": 128, "enable_rasteration": True},
        {"block_M": 64, "block_N": 64, "block_K": 64, "num_stages": 2, "thread_num": 128, "enable_rasteration": False},
        {"block_M": 64, "block_N": 128, "block_K": 32, "num_stages": 2, "thread_num": 128, "enable_rasteration": True},
        {"block_M": 64, "block_N": 128, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": False},
        {"block_M": 128, "block_N": 64, "block_K": 32, "num_stages": 1, "thread_num": 256, "enable_rasteration": True},
        {"block_M": 128, "block_N": 64, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": False},
        {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 2, "thread_num": 256, "enable_rasteration": True},
        {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": False},
        {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": False},
        {"block_M": -1, "block_N": 128, "block_K": 64, "num_stages": 1, "thread_num": 256, "enable_rasteration": True},
    ]


def test_default_selector_pool_k():
    assert SELECTOR.default_selector_pool_k(20) == 160
    assert SELECTOR.default_selector_pool_k(1) == 128
    assert SELECTOR.default_selector_pool_k(500) == 1024


def test_hopper_selector_is_deterministic_and_topk_bounded():
    configs = _candidate_grid()
    arch = _FakeArch(sm_version=90)
    selected_1, telemetry_1 = SELECTOR.select_configs(
        configs,
        topk=6,
        selector_name="hopper_hybrid_v1",
        arch=arch,
        m=4096,
        n=4096,
        k=4096,
        in_dtype="float16",
        accum_dtype="float32",
        selector_pool_k=256,
        selector_debug=False,
        gemm_like=True,
    )
    selected_2, telemetry_2 = SELECTOR.select_configs(
        configs,
        topk=6,
        selector_name="hopper_hybrid_v1",
        arch=arch,
        m=4096,
        n=4096,
        k=4096,
        in_dtype="float16",
        accum_dtype="float32",
        selector_pool_k=256,
        selector_debug=False,
        gemm_like=True,
    )

    assert len(selected_1) == 6
    assert selected_1 == selected_2
    assert telemetry_1.selector_name == "hopper_hybrid_v1"
    assert telemetry_1.selector_input_count == len(configs)
    assert telemetry_1.selector_output_count == 6
    assert telemetry_1.selector_fallback_used is False
    assert telemetry_2.selector_fallback_used is False


def test_hopper_selector_skips_non_hopper_arch():
    configs = _candidate_grid()
    selected, telemetry = SELECTOR.select_configs(
        configs,
        topk=4,
        selector_name="hopper_hybrid_v1",
        arch=_FakeArch(sm_version=80),
        m=4096,
        n=4096,
        k=4096,
        in_dtype="float16",
        accum_dtype="float32",
        selector_pool_k=None,
        selector_debug=False,
        gemm_like=True,
    )
    assert selected == configs
    assert telemetry.selector_name == "none"
    assert telemetry.selector_output_count == len(configs)

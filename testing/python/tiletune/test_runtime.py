import inspect
import pytest
from tilelang.autotuner import AutoTuner
from tilelang.tiletune import TileTuneConfig, check_compiler_resources
from tilelang.contrib.cuda_resource_info import KernelResourceUsage, parse_ptxas_output
from tilelang.tiletune.budget import resolve_register_budget
from tvm.target import Target


@pytest.mark.parametrize("arch", ["sm_50", "sm_70", "sm_80", "sm_89", "sm_90", "sm_90a", "sm_100a", "sm_110a", "sm_120f"])
@pytest.mark.parametrize("user_cap,expected", [(None, 255), (128, 128), (255, 255), (512, 255)])
def test_architecture_register_budget(arch, user_cap, expected):
    config = TileTuneConfig(register_cap=user_cap)
    budget = resolve_register_budget(config, {"kind": "cuda", "arch": arch})
    assert budget["budget"] == expected
    assert budget["hardware_register_cap"] == 255
    assert budget["target_arch"] == arch
    assert budget["budget_source"] == ("user register cap" if user_cap in (128, 255) else "architecture register limit")
    assert config.register_cap == user_cap


@pytest.mark.parametrize("target", [None, "cuda", {"kind": "cuda"}, "llvm", {"kind": "cuda", "arch": "sm_999"}])
def test_unknown_architecture_keeps_user_budget(target):
    budget = resolve_register_budget(TileTuneConfig(), target)
    assert budget["budget"] is None
    assert budget["hardware_register_cap"] is None
    assert budget["budget_source"] is None
    budget = resolve_register_budget(TileTuneConfig(register_cap=128), target)
    assert budget["budget"] == 128
    assert budget["budget_source"] == "user register cap"


def test_register_budget_uses_target_scope_without_overriding_explicit_target():
    with Target({"kind": "cuda", "arch": "sm_90a"}):
        assert resolve_register_budget(TileTuneConfig())["budget"] == 255
        assert resolve_register_budget(TileTuneConfig(), "llvm")["budget"] is None


def test_target_descriptions_do_not_query_a_device(monkeypatch):
    def unexpected_target_construction(*args, **kwargs):
        pytest.fail("budget resolution must not construct a target and implicitly query CUDA device 0")

    monkeypatch.setattr(Target, "__init__", unexpected_target_construction)
    config = TileTuneConfig()
    assert resolve_register_budget(config, "cuda")["budget"] is None
    assert resolve_register_budget(config, {"kind": "cuda"})["budget"] is None
    assert resolve_register_budget(config, {"kind": "cuda", "arch": "sm_90a"})["budget"] == 255
    assert resolve_register_budget(config, '{"kind": "cuda", "arch": "sm_90a"}')["budget"] == 255


def test_compiler_counters():
    missing = check_compiler_resources({}, ["kernel"])
    assert missing["keep"] and missing["status"] == "unknown"
    default = check_compiler_resources({"kernel": KernelResourceUsage()}, ["kernel"])
    assert all(x is None for x in default["resources"]["kernel"].values())
    parsed = parse_ptxas_output("""ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : Function properties for kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 255 registers
""")
    assert check_compiler_resources(parsed, ["kernel"])["status"] == "pass"
    parsed["kernel"].extra["spill_loads_bytes"] = 4
    assert not check_compiler_resources(parsed, ["kernel"])["keep"]
    assert check_compiler_resources(parsed, ["kernel"], {"mode": "report_only"})["keep"]
    assert check_compiler_resources(parsed, ["kernel"], {"max_spill_bytes": 4})["keep"]
    parsed["kernel"].local_size_bytes = 8
    assert not check_compiler_resources(parsed, ["kernel"], {"max_spill_bytes": 4})["keep"]
    assert check_compiler_resources(parsed, ["unrelated"])["status"] == "unknown"


@pytest.mark.parametrize("spill_cap,local_cap,reject", [(None, None, False), (0, None, True), (None, 0, True), (48, 48, False)])
def test_optional_spill_and_local_limits(spill_cap, local_cap, reject):
    resources = parse_ptxas_output("""ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : Function properties for kernel
    48 bytes stack frame, 48 bytes spill stores, 48 bytes spill loads
ptxas info : Used 168 registers
""")
    config = dict(max_spill_bytes=spill_cap, max_local_bytes=local_cap)
    decision = check_compiler_resources(resources, ["kernel"], config)
    assert decision["would_reject"] == reject
    assert decision["keep"] == (not reject)
    assert decision["resources"]["kernel"] == dict(registers=168, spill_stores_bytes=48, spill_loads_bytes=48, local_bytes=48)
    assert check_compiler_resources({}, ["kernel"], config)["status"] == "unknown"
    # Disabling spill/local limits does not disable an explicit register cap.
    assert not check_compiler_resources(resources, ["kernel"], dict(config, register_cap=128))["keep"]


@pytest.mark.parametrize(
    "n_regs,user_cap,mode,would_reject",
    [
        (255, None, "reject", False),
        (256, None, "reject", True),
        (256, 512, "reject", True),
        (129, 128, "reject", True),
        (256, None, "report_only", True),
    ],
)
def test_compiler_register_budget(n_regs, user_cap, mode, would_reject):
    resources = parse_ptxas_output(f"""ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : Function properties for kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used {n_regs} registers
""")
    decision = check_compiler_resources(
        resources, ["kernel"], {"register_cap": user_cap, "mode": mode}, target={"kind": "cuda", "arch": "sm_90a"}
    )
    assert decision["would_reject"] == would_reject
    assert decision["keep"] == (mode == "report_only" or not would_reject)
    assert decision["budget"] == (128 if user_cap == 128 else 255)
    assert decision["hardware_register_cap"] == 255
    missing = check_compiler_resources({}, ["kernel"], target={"kind": "cuda", "arch": "sm_90a"})
    assert missing["keep"] and missing["status"] == "unknown"


def kernel(block=32):
    return block


def test_settings_cache_identity():
    tuner = AutoTuner(kernel, [{"block": 32}])

    def key():
        return tuner.generate_cache_key(inspect.signature(kernel).parameters, {})

    disabled = key()
    tuner.set_tiletune_args(True)
    enabled = key()
    assert disabled != enabled
    tuner.set_tiletune_args(True, max_spill_bytes=None, max_local_bytes=None)
    assert key() != enabled
    tuner.set_tiletune_args(True, max_spill_bytes=0, max_local_bytes=0)
    assert key() == enabled
    tuner.set_tiletune_args(True, report_path="a.json")
    assert key() == enabled
    tuner.set_tiletune_args(True, trace_path="trace.log")
    assert key() == enabled
    tuner.set_tiletune_args(True, mode="report_only")
    assert key() != enabled
    tuner.set_compile_args(target={"kind": "cuda", "arch": "sm_90a"}, execution_backend="tvm_ffi")
    hopper = key()
    tuner.set_compile_args(target={"kind": "cuda", "arch": "sm_80"}, execution_backend="tvm_ffi")
    assert key() != hopper


@pytest.mark.parametrize("kwargs", [{"register_cap": 0}, {"max_spill_bytes": -1}, {"mode": "rank"}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TileTuneConfig(**kwargs)


def test_conflicts_and_hooks():
    tuner = AutoTuner(kernel, [{"block": 32}]).set_compile_args(target="cuda", execution_backend="tvm_ffi").set_tiletune_args(True)
    with pytest.raises(ValueError, match="early_stop=False"):
        tuner.run(early_stop=True)
    tuner.set_filter_args(True)
    with pytest.raises(ValueError, match="legacy"):
        tuner.run()
    tuner.set_filter_args(False)
    tuner.jit_compile = lambda **kwargs: None
    with pytest.raises(ValueError, match="opaque"):
        tuner.run()
    tuner.set_compile_args(target="llvm", execution_backend="tvm_ffi")
    with pytest.raises(ValueError, match="CUDA"):
        tuner.run()

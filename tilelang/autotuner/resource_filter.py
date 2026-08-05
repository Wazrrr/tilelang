"""Exact CUDA resource filtering for autotune candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tilelang import tvm

FilterStage = Literal["pre_compile", "post_compile"]
FilterVerdict = Literal["keep", "reject"]


@dataclass(frozen=True)
class AutotuneResourceFilterConfig:
    """Configuration for exact-only autotune resource filtering."""

    enabled: bool = False
    pre_compile: bool = True
    post_compile: bool = True
    device_id: int | None = None
    report_path: str | None = None

    @classmethod
    def from_value(cls, value: bool | dict[str, Any] | "AutotuneResourceFilterConfig" | None) -> "AutotuneResourceFilterConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported resource filter config: {type(value)!r}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["report_path"] is not None:
            data["report_path"] = str(data["report_path"])
        return data

    def to_cache_key_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("report_path", None)
        return data


@dataclass(frozen=True)
class LaunchResourceInfo:
    function_name: str
    block_dims: tuple[int | None, int | None, int | None] = (1, 1, 1)
    grid_dims: tuple[int | None, int | None, int | None] = (1, 1, 1)
    dynamic_smem_bytes: int | None = 0
    cluster_dims: tuple[int | None, int | None, int | None] | None = None
    uses_cooperative_groups: bool = False

    @property
    def threads_per_block(self) -> int | None:
        product = 1
        for dim in self.block_dims:
            if dim is None:
                return None
            product *= dim
        return product


@dataclass(frozen=True)
class CompiledResourceInfo:
    function_name: str
    n_regs: int = 0
    n_spills: int = 0
    n_max_threads: int | None = None
    static_smem_bytes: int = 0
    const_size_bytes: int = 0
    local_size_bytes: int = 0
    max_dynamic_smem_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CudaDeviceLimits:
    max_threads_per_block: int | None = None
    max_block_dims: tuple[int | None, int | None, int | None] = (None, None, None)
    max_grid_dims: tuple[int | None, int | None, int | None] = (None, None, None)
    max_shared_memory_per_block: int | None = None
    max_shared_memory_per_block_optin: int | None = None
    max_registers_per_block: int | None = None
    max_blocks_per_multiprocessor: int | None = None
    cooperative_launch: bool | None = None


@dataclass(frozen=True)
class AutotuneFilterDecision:
    verdict: FilterVerdict
    stage: FilterStage
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        return self.verdict == "keep"

    @classmethod
    def keep_decision(cls, stage: FilterStage, reason: str, **details: Any) -> "AutotuneFilterDecision":
        return cls("keep", stage, reason, details)

    @classmethod
    def reject_decision(cls, stage: FilterStage, reason: str, **details: Any) -> "AutotuneFilterDecision":
        return cls("reject", stage, reason, details)


class AutotuneResourceFilterReject(RuntimeError):
    """Internal marker for configs skipped by exact resource filtering."""

    def __init__(self, decision: AutotuneFilterDecision):
        self.decision = decision
        super().__init__(f"{decision.stage}:{decision.reason}:{decision.details}")


def query_cuda_device_limits(device_id: int | None = None) -> CudaDeviceLimits | None:
    try:
        import torch
        from tilelang.carver.arch import driver as cuda_driver

        if device_id is None:
            device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return CudaDeviceLimits(
            max_threads_per_block=cuda_driver.get_max_threads_per_block(device_id),
            max_block_dims=cuda_driver.get_max_block_dims(device_id),
            max_grid_dims=cuda_driver.get_max_grid_dims(device_id),
            max_shared_memory_per_block=cuda_driver.get_shared_memory_per_block(device_id),
            max_shared_memory_per_block_optin=cuda_driver.get_max_dynamic_shared_size_bytes(device_id),
            max_registers_per_block=cuda_driver.get_registers_per_block(device_id),
            max_blocks_per_multiprocessor=cuda_driver.get_max_blocks_per_multiprocessor(device_id),
            cooperative_launch=cuda_driver.get_cooperative_launch_support(device_id),
        )
    except Exception:
        return None


def extract_launch_resource_info(device_mod: tvm.IRModule) -> list[LaunchResourceInfo]:
    infos: list[LaunchResourceInfo] = []
    for global_var, func in device_mod.functions.items():
        attrs = getattr(func, "attrs", None) or {}
        function_name = str(attrs.get("global_symbol", getattr(global_var, "name_hint", str(global_var))))
        block_dims: list[int | None] = [1, 1, 1]
        grid_dims: list[int | None] = [1, 1, 1]

        if "thread_extent" in attrs:
            for tag, extent in attrs["thread_extent"].items():
                tag_str = str(tag)
                axis = tag_str[-1]
                if axis not in "xyz":
                    continue
                index = "xyz".index(axis)
                if "threadIdx" in tag_str:
                    block_dims[index] = _exact_int(extent)
                elif "blockIdx" in tag_str:
                    grid_dims[index] = _exact_int(extent)

        cluster_dims = None
        if "cluster_dims" in attrs:
            raw_cluster_dims = attrs["cluster_dims"]
            parsed = [_exact_int(raw_cluster_dims[i]) for i in range(len(raw_cluster_dims))]
            cluster_dims = tuple((parsed + [1, 1, 1])[:3])

        dynamic_smem_bytes = _exact_int(attrs["dyn_shared_memory_buf"]) if "dyn_shared_memory_buf" in attrs else 0
        infos.append(
            LaunchResourceInfo(
                function_name=function_name,
                block_dims=tuple(block_dims),
                grid_dims=tuple(grid_dims),
                dynamic_smem_bytes=dynamic_smem_bytes,
                cluster_dims=cluster_dims,
                uses_cooperative_groups=bool(attrs.get("use_cooperative_groups", False)),
            )
        )
    return infos


def evaluate_pre_compile_resource_filter(
    launch_infos: list[LaunchResourceInfo],
    limits: CudaDeviceLimits | None,
) -> AutotuneFilterDecision:
    if limits is None:
        return AutotuneFilterDecision.keep_decision("pre_compile", "cuda_device_limits_unavailable")

    for info in launch_infos:
        threads_per_block = info.threads_per_block
        if (
            threads_per_block is not None
            and limits.max_threads_per_block is not None
            and threads_per_block > limits.max_threads_per_block
        ):
            return AutotuneFilterDecision.reject_decision(
                "pre_compile",
                "threads_per_block_over_limit",
                function=info.function_name,
                threads_per_block=threads_per_block,
                limit=limits.max_threads_per_block,
            )

        dim_decision = _check_dims("block_dim_over_limit", info.function_name, info.block_dims, limits.max_block_dims)
        if dim_decision is not None:
            return dim_decision
        dim_decision = _check_dims("grid_dim_over_limit", info.function_name, info.grid_dims, limits.max_grid_dims)
        if dim_decision is not None:
            return dim_decision

        smem_limit = limits.max_shared_memory_per_block_optin or limits.max_shared_memory_per_block
        if info.dynamic_smem_bytes is not None and smem_limit is not None and info.dynamic_smem_bytes > smem_limit:
            return AutotuneFilterDecision.reject_decision(
                "pre_compile",
                "dynamic_shared_memory_over_limit",
                function=info.function_name,
                dynamic_smem_bytes=info.dynamic_smem_bytes,
                limit=smem_limit,
            )

        if info.uses_cooperative_groups and limits.cooperative_launch is False:
            return AutotuneFilterDecision.reject_decision(
                "pre_compile",
                "cooperative_launch_not_supported",
                function=info.function_name,
            )

    return AutotuneFilterDecision.keep_decision("pre_compile", "resource_fit_or_unknown")


def compiled_resource_info_from_usage(function_name: str, usage: Any) -> CompiledResourceInfo:
    return CompiledResourceInfo(
        function_name=function_name,
        n_regs=int(getattr(usage, "n_regs", 0) or 0),
        n_spills=int(getattr(usage, "n_spills", 0) or 0),
        n_max_threads=getattr(usage, "n_max_threads", None),
        static_smem_bytes=int(getattr(usage, "static_smem_bytes", 0) or 0),
        const_size_bytes=int(getattr(usage, "const_size_bytes", 0) or 0),
        local_size_bytes=int(getattr(usage, "local_size_bytes", 0) or 0),
        max_dynamic_smem_bytes=getattr(usage, "max_dynamic_smem_bytes", None),
        extra=dict(getattr(usage, "extra", {}) or {}),
    )


def evaluate_post_compile_resource_filter(
    launch_infos: list[LaunchResourceInfo],
    resource_usage: dict[str, Any],
    limits: CudaDeviceLimits | None,
) -> AutotuneFilterDecision:
    if limits is None:
        return AutotuneFilterDecision.keep_decision("post_compile", "cuda_device_limits_unavailable")
    if not resource_usage:
        return AutotuneFilterDecision.keep_decision("post_compile", "compiled_resource_usage_unavailable")

    for launch in launch_infos:
        raw_usage = resource_usage.get(launch.function_name)
        if raw_usage is None:
            continue
        compiled = compiled_resource_info_from_usage(launch.function_name, raw_usage)
        threads_per_block = launch.threads_per_block

        if compiled.n_max_threads is not None and threads_per_block is not None and threads_per_block > compiled.n_max_threads:
            return AutotuneFilterDecision.reject_decision(
                "post_compile",
                "threads_per_block_over_compiled_function_limit",
                function=launch.function_name,
                threads_per_block=threads_per_block,
                limit=compiled.n_max_threads,
            )

        if (
            limits.max_registers_per_block is not None
            and compiled.n_regs > 0
            and threads_per_block is not None
            and compiled.n_regs * threads_per_block > limits.max_registers_per_block
        ):
            return AutotuneFilterDecision.reject_decision(
                "post_compile",
                "registers_per_block_over_limit",
                function=launch.function_name,
                registers_per_thread=compiled.n_regs,
                threads_per_block=threads_per_block,
                registers_per_block=compiled.n_regs * threads_per_block,
                limit=limits.max_registers_per_block,
            )

        dynamic_smem = launch.dynamic_smem_bytes
        if (
            dynamic_smem is not None
            and compiled.max_dynamic_smem_bytes is not None
            and dynamic_smem > compiled.max_dynamic_smem_bytes
        ):
            return AutotuneFilterDecision.reject_decision(
                "post_compile",
                "dynamic_shared_memory_over_compiled_function_limit",
                function=launch.function_name,
                dynamic_smem_bytes=dynamic_smem,
                limit=compiled.max_dynamic_smem_bytes,
            )

        block_smem_limit = limits.max_shared_memory_per_block_optin or limits.max_shared_memory_per_block
        if dynamic_smem is not None and block_smem_limit is not None:
            total_smem = compiled.static_smem_bytes + dynamic_smem
            if total_smem > block_smem_limit:
                return AutotuneFilterDecision.reject_decision(
                    "post_compile",
                    "total_shared_memory_over_limit",
                    function=launch.function_name,
                    static_smem_bytes=compiled.static_smem_bytes,
                    dynamic_smem_bytes=dynamic_smem,
                    total_smem_bytes=total_smem,
                    limit=block_smem_limit,
                )

    return AutotuneFilterDecision.keep_decision("post_compile", "resource_fit_or_unknown")


def _check_dims(
    reason: str,
    function_name: str,
    dims: tuple[int | None, int | None, int | None],
    limits: tuple[int | None, int | None, int | None],
) -> AutotuneFilterDecision | None:
    for axis, (dim, limit) in enumerate(zip(dims, limits)):
        if dim is None or limit is None:
            continue
        if dim > limit:
            return AutotuneFilterDecision.reject_decision(
                "pre_compile",
                reason,
                function=function_name,
                axis="xyz"[axis],
                dim=dim,
                limit=limit,
            )
    return None


def _exact_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if hasattr(value, "value"):
        raw_value = value.value
        if isinstance(raw_value, (int, bool)):
            return int(raw_value)
    try:
        if isinstance(value, tvm.tirx.IntImm):
            return int(value)
    except Exception:
        pass
    return None

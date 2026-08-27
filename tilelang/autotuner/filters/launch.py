"""Launch metadata extraction used by autotune verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tilelang import tvm


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

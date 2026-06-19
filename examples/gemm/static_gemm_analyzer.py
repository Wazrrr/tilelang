from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tilelang as tl
from tilelang.transform import PassConfigKey


_DTYPE_BITS = {
    "bool": 1,
    "int1": 1,
    "int2": 2,
    "int4": 4,
    "float4_e2m1fn": 4,
    "float4_e2m1fnx2": 4,
    "float6_e2m3fn": 6,
    "float6_e3m2fn": 6,
    "int8": 8,
    "uint8": 8,
    "float8_e4m3": 8,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e5m2fnuz": 8,
    "float16": 16,
    "half": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "float32": 32,
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("T.", "").replace("tl.", "").strip("'\"")


def dtype_bits(dtype: Any) -> int:
    name = _dtype_name(dtype)
    if name in _DTYPE_BITS:
        return _DTYPE_BITS[name]
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unsupported dtype for static GEMM analysis: {dtype!r}")


def dtype_nbytes(dtype: Any) -> int:
    return max(1, (dtype_bits(dtype) + 7) // 8)


@dataclass(frozen=True)
class GemmConfig:
    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    thread_num: int
    enable_rasteration: bool
    dtype: str = "bfloat16"
    accum_dtype: str = "float32"

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        *,
        dtype: Any = "bfloat16",
        accum_dtype: Any = "float32",
    ) -> "GemmConfig":
        return cls(
            block_M=int(config["block_M"]),
            block_N=int(config["block_N"]),
            block_K=int(config["block_K"]),
            num_stages=int(config["num_stages"]),
            thread_num=int(config["thread_num"]),
            enable_rasteration=bool(config["enable_rasteration"]),
            dtype=_dtype_name(dtype),
            accum_dtype=_dtype_name(accum_dtype),
        )

    def autotune_dict(self) -> dict[str, Any]:
        return {
            "block_M": self.block_M,
            "block_N": self.block_N,
            "block_K": self.block_K,
            "num_stages": self.num_stages,
            "thread_num": self.thread_num,
            "enable_rasteration": self.enable_rasteration,
        }


@dataclass(frozen=True)
class DeviceLimits:
    name: str = "unknown"
    sm_version: int | None = None
    sm_count: int | None = None
    warp_size: int = 32
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    shared_memory_per_block: int = 0
    shared_memory_per_sm: int = 0
    registers_per_block: int = 65536
    registers_per_sm: int = 65536


@dataclass(frozen=True)
class PtxasResourceInfo:
    registers_per_thread: int | None = None
    smem_bytes: int | None = None
    cmem_bytes: int | None = None
    stack_frame_bytes: int | None = None
    spill_stores_bytes: int | None = None
    spill_loads_bytes: int | None = None
    raw_output: str = ""

    @property
    def spill_bytes(self) -> int:
        return (self.spill_stores_bytes or 0) + (self.spill_loads_bytes or 0)

    @property
    def has_metadata(self) -> bool:
        return any(
            value is not None
            for value in (
                self.registers_per_thread,
                self.smem_bytes,
                self.stack_frame_bytes,
                self.spill_stores_bytes,
                self.spill_loads_bytes,
            )
        )


@dataclass(frozen=True)
class SourceSummary:
    launch_bounds_threads: int | None = None
    global_kernel_names: list[str] = field(default_factory=list)
    wgmma_ops: int = 0
    mma_ops: int = 0
    cp_async_ops: int = 0
    tma_ops: int = 0


@dataclass(frozen=True)
class StaticEstimate:
    tile_flops_per_cta: int
    global_bytes_per_cta: int
    arithmetic_intensity: float
    shared_bytes_estimate: int
    pipeline_shared_bytes_estimate: int
    accumulator_registers_per_thread_estimate: int
    registers_per_thread_estimate: int
    k_iterations: int


@dataclass(frozen=True)
class GemmResourceReport:
    config: dict[str, Any]
    verdict: str
    reasons: list[str]
    score: float
    device: DeviceLimits
    static: StaticEstimate
    source: SourceSummary
    ptxas: PtxasResourceInfo
    active_blocks_per_sm_estimate: int
    occupancy_limiters: dict[str, int | None]
    compile_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GemmAnalysisSummary:
    reports: list[GemmResourceReport]
    original_count: int
    topk: int | None = None

    @property
    def kept_reports(self) -> list[GemmResourceReport]:
        return [report for report in self.reports if report.verdict == "keep"]

    @property
    def rejected_reports(self) -> list[GemmResourceReport]:
        return [report for report in self.reports if report.verdict == "reject"]

    def ranked_kept_reports(self) -> list[GemmResourceReport]:
        return sorted(self.kept_reports, key=lambda report: report.score, reverse=True)

    def selected_reports(self) -> list[GemmResourceReport]:
        ranked = self.ranked_kept_reports()
        if self.topk is None or self.topk <= 0:
            return ranked
        return ranked[: self.topk]

    def selected_configs(self) -> list[dict[str, Any]]:
        return [report.config for report in self.selected_reports()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_count": self.original_count,
            "kept_count": len(self.kept_reports),
            "rejected_count": len(self.rejected_reports),
            "selected_count": len(self.selected_reports()),
            "topk": self.topk,
            "reports": [report.to_dict() for report in self.reports],
        }


def parse_ptxas_output(output: str) -> PtxasResourceInfo:
    registers = _max_int_match(r"Used\s+(\d+)\s+registers", output)
    stack = _max_int_match(r"(\d+)\s+bytes\s+stack\s+frame", output)
    spill_stores = _max_int_match(r"(\d+)\s+bytes\s+spill\s+stores", output)
    spill_loads = _max_int_match(r"(\d+)\s+bytes\s+spill\s+loads", output)
    smem = _max_named_byte_value(output, "smem")
    cmem_values = [_max_named_byte_value(output, f"cmem[{idx}]") for idx in range(16)]
    cmem_values = [value for value in cmem_values if value is not None]
    cmem = max(cmem_values) if cmem_values else None
    return PtxasResourceInfo(
        registers_per_thread=registers,
        smem_bytes=smem,
        cmem_bytes=cmem,
        stack_frame_bytes=stack,
        spill_stores_bytes=spill_stores,
        spill_loads_bytes=spill_loads,
        raw_output=output,
    )


def parse_cuda_source(source: str) -> SourceSummary:
    launch_bounds = _max_int_match(r"__launch_bounds__\(\s*(\d+)", source)
    kernel_names = []
    seen = set()
    kernel_pattern = re.compile(r'(?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+)')
    for match in kernel_pattern.finditer(source):
        name = match.group(1)
        if name not in seen:
            kernel_names.append(name)
            seen.add(name)
    return SourceSummary(
        launch_bounds_threads=launch_bounds,
        global_kernel_names=kernel_names,
        wgmma_ops=source.count("wgmma"),
        mma_ops=source.count("mma_sync") + source.count("mma."),
        cp_async_ops=source.count("cp.async"),
        tma_ops=source.count("tma_") + source.count("CUtensorMap"),
    )


def analyze_cuda_source(
    source: str,
    config: GemmConfig | dict[str, Any],
    arch: Any,
    *,
    M: int,
    N: int,
    K: int,
    ptxas_output: str = "",
    compile_error: str | None = None,
) -> GemmResourceReport:
    gemm_config = config if isinstance(config, GemmConfig) else GemmConfig.from_dict(config)
    device = device_limits_from_arch(arch)
    static = estimate_static_resources(gemm_config, M=M, N=N, K=K)
    source_summary = parse_cuda_source(source)
    ptxas = parse_ptxas_output(ptxas_output)
    verdict, reasons, active_blocks, limiters = classify_config(
        gemm_config,
        device,
        static,
        ptxas,
        compile_error=compile_error,
    )
    score = score_config(gemm_config, device, static, ptxas, active_blocks, reasons, verdict)
    return GemmResourceReport(
        config=gemm_config.autotune_dict(),
        verdict=verdict,
        reasons=reasons,
        score=score,
        device=device,
        static=static,
        source=source_summary,
        ptxas=ptxas,
        active_blocks_per_sm_estimate=active_blocks,
        occupancy_limiters=limiters,
        compile_error=compile_error,
    )


def analyze_tilelang_gemm(
    kernel_builder: Any,
    config: GemmConfig | dict[str, Any],
    arch: Any,
    *,
    M: int,
    N: int,
    K: int,
    target: str = "auto",
    execution_backend: str = "tvm_ffi",
    out_idx: list[int] | int | None = None,
    pass_configs: dict[str, Any] | None = None,
    use_cache: bool = False,
) -> GemmResourceReport:
    gemm_config = config if isinstance(config, GemmConfig) else GemmConfig.from_dict(config)
    compile_pass_configs = dict(pass_configs or {})
    compile_pass_configs[PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT] = True

    source = ""
    compile_log = ""
    compile_error = None
    stream = io.StringIO()
    try:
        prim_func = kernel_builder(**gemm_config.autotune_dict())
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with _tilelang_cache_setting(enabled=use_cache):
                compiled = tl.compile(
                    prim_func,
                    out_idx=out_idx,
                    target=target,
                    execution_backend=execution_backend,
                    pass_configs=compile_pass_configs,
                )
        source = compiled.get_kernel_source() or ""
    except Exception as exc:  # pylint: disable=broad-except
        compile_error = _truncate_text(str(exc), 4000)
    finally:
        compile_log = stream.getvalue()

    return analyze_cuda_source(
        source,
        gemm_config,
        arch,
        M=M,
        N=N,
        K=K,
        ptxas_output=compile_log,
        compile_error=compile_error,
    )


def analyze_config_space(
    kernel_builder: Any,
    configs: list[dict[str, Any]],
    arch: Any,
    *,
    M: int,
    N: int,
    K: int,
    topk: int | None = None,
    target: str = "auto",
    execution_backend: str = "tvm_ffi",
    out_idx: list[int] | int | None = None,
    pass_configs: dict[str, Any] | None = None,
    use_cache: bool = False,
    verbose: bool = True,
) -> GemmAnalysisSummary:
    reports: list[GemmResourceReport] = []
    total = len(configs)
    for idx, config in enumerate(configs, start=1):
        if verbose:
            print(f"[static-gemm-analyzer] analyzing {idx}/{total}: {config}")
        reports.append(
            analyze_tilelang_gemm(
                kernel_builder,
                config,
                arch,
                M=M,
                N=N,
                K=K,
                target=target,
                execution_backend=execution_backend,
                out_idx=out_idx,
                pass_configs=pass_configs,
                use_cache=use_cache,
            )
        )
    return GemmAnalysisSummary(reports=reports, original_count=total, topk=topk)


def estimate_static_resources(config: GemmConfig, *, M: int, N: int, K: int) -> StaticEstimate:
    dtype_bytes = dtype_nbytes(config.dtype)
    accum_bits = dtype_bits(config.accum_dtype)
    output_bytes = dtype_bytes
    k_iterations = max(1, math.ceil(K / config.block_K))

    a_shared = config.block_M * config.block_K * dtype_bytes
    b_shared = config.block_N * config.block_K * dtype_bytes
    c_shared = config.block_M * config.block_N * output_bytes
    stage_multiplier = max(1, config.num_stages)
    pipeline_shared = (a_shared + b_shared) * stage_multiplier
    shared_bytes = pipeline_shared + c_shared

    tile_flops_per_cta = 2 * config.block_M * config.block_N * min(K, config.block_K * k_iterations)
    global_load_bytes = k_iterations * (a_shared + b_shared)
    global_store_bytes = c_shared
    global_bytes = global_load_bytes + global_store_bytes
    arithmetic_intensity = tile_flops_per_cta / global_bytes if global_bytes else 0.0

    accum_regs = math.ceil(config.block_M * config.block_N * accum_bits / (32 * max(1, config.thread_num)))
    # Extra per-thread registers cover address arithmetic, predicates, descriptors and operand fragments.
    estimated_regs = accum_regs + 32
    return StaticEstimate(
        tile_flops_per_cta=tile_flops_per_cta,
        global_bytes_per_cta=global_bytes,
        arithmetic_intensity=arithmetic_intensity,
        shared_bytes_estimate=shared_bytes,
        pipeline_shared_bytes_estimate=pipeline_shared,
        accumulator_registers_per_thread_estimate=accum_regs,
        registers_per_thread_estimate=estimated_regs,
        k_iterations=k_iterations,
    )


def classify_config(
    config: GemmConfig,
    device: DeviceLimits,
    static: StaticEstimate,
    ptxas: PtxasResourceInfo,
    *,
    compile_error: str | None = None,
) -> tuple[str, list[str], int, dict[str, int | None]]:
    reasons: list[str] = []
    if compile_error is not None:
        reasons.append("compile_failed")

    register_source = ptxas.registers_per_thread or static.registers_per_thread_estimate
    register_bytes = register_source * config.thread_num
    shared_for_limits = max(static.shared_bytes_estimate, ptxas.smem_bytes or 0)

    shared_limit = _positive_or_none(device.shared_memory_per_block)
    if shared_limit is not None and shared_for_limits > shared_limit:
        reasons.append("shared_memory_over_cap")

    reg_limit = _positive_or_none(device.registers_per_block)
    if reg_limit is not None and register_bytes > reg_limit:
        if ptxas.registers_per_thread is not None:
            reasons.append("registers_over_cap")
        else:
            reasons.append("register_estimate_over_cap")

    if ptxas.spill_bytes > 0:
        reasons.append("register_spill_risk")

    if config.thread_num % max(1, device.warp_size) != 0:
        reasons.append("non_warp_aligned_threads")
    if config.block_M % 16 != 0 or config.block_N % 16 != 0 or config.block_K % 16 != 0:
        reasons.append("low_tensorcore_alignment")
    if config.num_stages > 3:
        reasons.append("stage_overhead_risk")
    if static.arithmetic_intensity < 32:
        reasons.append("low_arithmetic_intensity")

    limiters = _estimate_active_blocks_per_sm(config, device, static, ptxas)
    positive_limits = [value for value in limiters.values() if value is not None]
    active_blocks = min(positive_limits) if positive_limits else 1
    if active_blocks <= 0:
        reasons.append("zero_estimated_occupancy")

    hard_reasons = {
        "compile_failed",
        "shared_memory_over_cap",
        "registers_over_cap",
        "register_estimate_over_cap",
        "register_spill_risk",
        "zero_estimated_occupancy",
    }
    verdict = "reject" if any(reason in hard_reasons for reason in reasons) else "keep"
    if verdict == "keep" and not reasons:
        reasons.append("resource_fit")
    return verdict, reasons, max(0, active_blocks), limiters


def score_config(
    config: GemmConfig,
    device: DeviceLimits,
    static: StaticEstimate,
    ptxas: PtxasResourceInfo,
    active_blocks: int,
    reasons: list[str],
    verdict: str,
) -> float:
    if verdict == "reject":
        return -1_000_000.0 - len(reasons)
    registers = ptxas.registers_per_thread or static.registers_per_thread_estimate
    shared_for_limits = max(static.shared_bytes_estimate, ptxas.smem_bytes or 0)
    smem_headroom = 0.0
    if device.shared_memory_per_block > 0:
        smem_headroom = max(0.0, 1.0 - shared_for_limits / device.shared_memory_per_block)

    score = 0.0
    score += active_blocks * 100.0
    score += min(static.arithmetic_intensity, 256.0) * 0.5
    score += smem_headroom * 25.0
    score -= registers * 0.25
    score -= max(0, config.num_stages - 2) * 2.0
    score -= len([reason for reason in reasons if reason != "resource_fit"]) * 5.0
    return score


def device_limits_from_arch(arch: Any) -> DeviceLimits:
    target = getattr(arch, "target", None)
    target_attrs = getattr(target, "attrs", {}) or {}

    name = str(getattr(arch, "name", "unknown"))
    sm_version = _safe_int(getattr(arch, "sm_version", None))
    if sm_version is None:
        target_arch = str(target_attrs.get("arch", ""))
        match = re.search(r"sm_?(\d+)", target_arch)
        sm_version = int(match.group(1)) if match else None

    sm_count = _safe_int(getattr(arch, "compute_max_core", None))
    warp_size = _safe_int(getattr(arch, "warp_size", None)) or 32
    smem_cap = _max_positive(
        getattr(arch, "smem_cap", None),
        getattr(arch, "max_smem_usage", None),
        target_attrs.get("max_shared_memory_per_block", None),
    )
    regs = _max_positive(getattr(arch, "reg_cap", None), target_attrs.get("registers_per_block", None), 65536)
    max_threads_per_sm = 2048

    should_query_cuda_runtime = getattr(arch, "platform", None) == "CUDA" or arch.__class__.__name__ == "CUDA"
    if should_query_cuda_runtime:
        try:
            from tilelang.carver.arch.driver import cuda_driver

            props = cuda_driver.get_cuda_device_properties()
            if props is not None:
                name = str(getattr(props, "name", name) or name)
                sm_count = _safe_int(getattr(props, "multi_processor_count", None)) or sm_count
                max_threads_per_sm = _safe_int(getattr(props, "max_threads_per_multi_processor", None)) or max_threads_per_sm
            smem_cap = _max_positive(smem_cap, cuda_driver.get_max_dynamic_shared_size_bytes())
            regs = _max_positive(regs, cuda_driver.get_registers_per_block())
        except Exception:
            pass

    smem_per_block = smem_cap or 0
    smem_per_sm = smem_cap or smem_per_block
    return DeviceLimits(
        name=name,
        sm_version=sm_version,
        sm_count=sm_count,
        warp_size=warp_size,
        max_threads_per_sm=max_threads_per_sm,
        max_blocks_per_sm=32,
        shared_memory_per_block=smem_per_block,
        shared_memory_per_sm=smem_per_sm,
        registers_per_block=regs or 65536,
        registers_per_sm=regs or 65536,
    )


def write_static_report(summary: GemmAnalysisSummary, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".tsv":
        rows = [_flatten_report(report) for report in summary.reports]
        fieldnames = list(rows[0].keys()) if rows else ["config", "verdict", "score", "reasons"]
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return

    with output_path.open("w") as file:
        json.dump(summary.to_dict(), file, indent=2, sort_keys=True)


def _flatten_report(report: GemmResourceReport) -> dict[str, Any]:
    return {
        "config": json.dumps(report.config, sort_keys=True),
        "verdict": report.verdict,
        "score": report.score,
        "reasons": ",".join(report.reasons),
        "active_blocks_per_sm_estimate": report.active_blocks_per_sm_estimate,
        "static_shared_bytes": report.static.shared_bytes_estimate,
        "static_regs_per_thread": report.static.registers_per_thread_estimate,
        "arithmetic_intensity": report.static.arithmetic_intensity,
        "ptxas_registers_per_thread": report.ptxas.registers_per_thread,
        "ptxas_smem_bytes": report.ptxas.smem_bytes,
        "ptxas_spill_bytes": report.ptxas.spill_bytes,
        "compile_error": report.compile_error or "",
    }


def _estimate_active_blocks_per_sm(
    config: GemmConfig,
    device: DeviceLimits,
    static: StaticEstimate,
    ptxas: PtxasResourceInfo,
) -> dict[str, int | None]:
    shared_for_limits = max(static.shared_bytes_estimate, ptxas.smem_bytes or 0)
    regs_per_thread = ptxas.registers_per_thread or static.registers_per_thread_estimate
    regs_per_block = regs_per_thread * config.thread_num

    thread_limit = device.max_threads_per_sm // config.thread_num if config.thread_num > 0 else 0
    smem_limit = None
    if device.shared_memory_per_sm > 0:
        smem_limit = device.shared_memory_per_sm // shared_for_limits if shared_for_limits > 0 else device.max_blocks_per_sm
    reg_limit = None
    if device.registers_per_sm > 0:
        reg_limit = device.registers_per_sm // regs_per_block if regs_per_block > 0 else device.max_blocks_per_sm
    return {
        "threads": thread_limit,
        "shared_memory": smem_limit,
        "registers": reg_limit,
        "blocks": device.max_blocks_per_sm,
    }


def _max_int_match(pattern: str, text: str) -> int | None:
    matches = [int(match.group(1)) for match in re.finditer(pattern, text)]
    return max(matches) if matches else None


def _max_named_byte_value(text: str, name: str) -> int | None:
    escaped = re.escape(name)
    return _max_int_match(rf"(\d+)\s+bytes\s+{escaped}", text)


def _max_positive(*values: Any) -> int | None:
    positives = [_safe_int(value) for value in values]
    positives = [value for value in positives if value is not None and value > 0]
    return max(positives) if positives else None


def _positive_or_none(value: int | None) -> int | None:
    return value if value is not None and value > 0 else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... truncated ..."


@contextlib.contextmanager
def _tilelang_cache_setting(enabled: bool):
    is_cache_enabled = getattr(tl, "is_cache_enabled", None)
    enable_cache = getattr(tl, "enable_cache", None)
    disable_cache = getattr(tl, "disable_cache", None)
    if not callable(is_cache_enabled) or not callable(enable_cache) or not callable(disable_cache):
        yield
        return

    previous = bool(is_cache_enabled())
    if previous == enabled:
        yield
        return

    if enabled:
        enable_cache()
    else:
        disable_cache()
    try:
        yield
    finally:
        if previous:
            enable_cache()
        else:
            disable_cache()

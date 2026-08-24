from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tilelang.autotuner.filters import (
    AutotuneFilterDecision,
    AutotuneResourceFilterConfig,
    CudaDeviceLimits,
    LaunchResourceInfo,
    evaluate_post_compile_resource_filter,
    evaluate_pre_compile_resource_filter,
)
from tilelang.contrib.cuda_resource_info import KernelResourceUsage, parse_ptxas_output


@dataclass(frozen=True)
class GemmResourceReport:
    config: dict[str, Any]
    verdict: str
    reason: str
    details: dict[str, Any]

    @classmethod
    def from_decision(cls, config: dict[str, Any], decision: AutotuneFilterDecision) -> "GemmResourceReport":
        return cls(
            config=dict(config),
            verdict=decision.verdict,
            reason=decision.reason,
            details=dict(decision.details),
        )


@dataclass(frozen=True)
class GemmAnalysisSummary:
    reports: list[GemmResourceReport]

    @property
    def kept_reports(self) -> list[GemmResourceReport]:
        return [report for report in self.reports if report.verdict == "keep"]

    @property
    def rejected_reports(self) -> list[GemmResourceReport]:
        return [report for report in self.reports if report.verdict == "reject"]

    def selected_configs(self) -> list[dict[str, Any]]:
        return [report.config for report in self.kept_reports]


def estimate_registers_from_device_code(source: str) -> int:
    del source
    raise RuntimeError("Exact register usage is available only after NVCC/PTXAS compilation.")


def analyze_launch_resources(
    config: dict[str, Any],
    launch_infos: list[LaunchResourceInfo],
    limits: CudaDeviceLimits,
) -> GemmResourceReport:
    decision = evaluate_pre_compile_resource_filter(launch_infos, limits)
    return GemmResourceReport.from_decision(config, decision)


def analyze_compiled_resources(
    config: dict[str, Any],
    launch_infos: list[LaunchResourceInfo],
    ptxas_output: str,
    limits: CudaDeviceLimits,
) -> GemmResourceReport:
    usage = parse_ptxas_output(ptxas_output)
    decision = evaluate_post_compile_resource_filter(launch_infos, usage, limits)
    return GemmResourceReport.from_decision(config, decision)


def analyze_config_space(
    configs: list[dict[str, Any]],
    launch_infos_by_config: list[list[LaunchResourceInfo]],
    limits: CudaDeviceLimits,
) -> GemmAnalysisSummary:
    reports = [
        analyze_launch_resources(config, launch_infos, limits)
        for config, launch_infos in zip(configs, launch_infos_by_config)
    ]
    return GemmAnalysisSummary(reports=reports)


def write_static_report(summary: GemmAnalysisSummary, path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total": len(summary.reports),
        "kept": len(summary.kept_reports),
        "rejected": len(summary.rejected_reports),
        "reports": [
            {
                "config": report.config,
                "verdict": report.verdict,
                "reason": report.reason,
                "details": report.details,
            }
            for report in summary.reports
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "AutotuneResourceFilterConfig",
    "CudaDeviceLimits",
    "GemmAnalysisSummary",
    "GemmResourceReport",
    "KernelResourceUsage",
    "LaunchResourceInfo",
    "analyze_compiled_resources",
    "analyze_config_space",
    "analyze_launch_resources",
    "estimate_registers_from_device_code",
    "parse_ptxas_output",
    "write_static_report",
]

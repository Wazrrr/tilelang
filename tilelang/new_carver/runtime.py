"""Pressure-only compiler checks and per-config reports."""

from contextlib import contextmanager
from pathlib import Path
import json
import time

from tilelang.utils.autotune_timing import timed_autotune_stage
from .analysis import analyze_prim_func
from .config import ANALYSIS_VERSION, CarverConfig, CarverReject
from .budget import resolve_register_budget
from .ranking import rank_records


def check_compiler_resources(resource_usage, function_names, config=None, *, target=None, launch_infos=None, device_limits=None):
    """Match exact device symbols; absent counters remain unknown.

    The legacy recorder's default zeros are not observations. Its optional
    ``observed_fields`` marker distinguishes a reported zero from a default.
    """
    config = CarverConfig.from_value(config)
    register_budget = resolve_register_budget(config, target)
    launches = {info.function_name: info for info in launch_infos or []}
    physical = {}
    sm_registers = (device_limits or config.device_limits or {}).get("registers_per_sm")
    resources, reasons = {}, []
    unknown = not function_names
    for name in function_names:
        item = resource_usage.get(name)
        counters = dict(registers=None, spill_stores_bytes=None, spill_loads_bytes=None, local_bytes=None)
        if item is not None:
            extra = item.extra
            observed = extra.get("observed_fields", [])
            counters.update(
                registers=item.n_regs if item.n_regs or "n_regs" in observed else None,
                spill_stores_bytes=extra.get("spill_stores_bytes"),
                spill_loads_bytes=extra.get("spill_loads_bytes"),
                local_bytes=item.local_size_bytes if item.local_size_bytes or "local_size_bytes" in observed else None,
            )
        resources[name] = counters
        if launch_infos is not None:
            info = launches.get(name)
            threads = info.threads_per_block if info is not None else None
            # PTXAS reports initial per-thread allocation. Rounding to physical
            # allocation granularity can only increase this CTA lower bound.
            allocation = counters["registers"] * threads if counters["registers"] is not None and threads else None
            exceeds = allocation is not None and sm_registers is not None and allocation > sm_registers
            physical[name] = {
                "initial_registers_per_block_lower_bound": allocation,
                "launch_threads": threads,
                "registers_per_sm": sm_registers,
                "status": "exceeds_limit"
                if exceeds
                else "within_unrounded_limit"
                if allocation is not None and sm_registers
                else "unknown",
                "assumptions": ["allocation granularity and dynamic redistribution require additional occupancy modeling"],
            }
            unknown |= physical[name]["status"] == "unknown"
            if exceeds:
                reasons.append(f"{name}: initial CTA register allocation {allocation} exceeds SM capacity {sm_registers}")
        unknown |= any(value is None for value in counters.values())
        for field, cap in (
            ("spill_stores_bytes", config.max_spill_bytes),
            ("spill_loads_bytes", config.max_spill_bytes),
            ("local_bytes", config.max_local_bytes),
            ("registers", register_budget["budget"]),
        ):
            if cap is not None and counters[field] is not None and counters[field] > cap:
                reasons.append(f"{name}: {field}={counters[field]} exceeds {cap}")
    return {
        "keep": not reasons or config.mode == "report_only",
        "would_reject": bool(reasons),
        "status": "reject" if reasons else "unknown" if unknown else "pass",
        "reasons": reasons,
        "resources": resources,
        "physical_register_allocation": physical,
        **register_budget,
    }


class CarverSession:
    def __init__(self, config, configs, compile_flags=None, *, target=None, device_limits=None):
        self.config = config
        self.target = target
        self.device_limits = device_limits or config.device_limits
        self.compile_flags = list(compile_flags or [])
        self.started = time.perf_counter()
        self.records = [
            {
                "index": i,
                "config": dict(c),
                "analysis_version": ANALYSIS_VERSION,
                "propagation": None,
                "pressure": None,
                "compiler_resources": None,
                "pre_lowering": None,
                "post_compile": None,
                "status": "pending",
                "timings_ms": {},
            }
            for i, c in enumerate(configs)
        ]

    @contextmanager
    def stage(self, idx, name):
        start = time.perf_counter()
        try:
            with timed_autotune_stage(f"new_carver.{name}", config_idx=idx):
                yield
        finally:
            timing = self.records[idx]["timings_ms"]
            timing[name] = timing.get(name, 0) + (time.perf_counter() - start) * 1000

    def elaborate(self, idx, config_arg, elaborate_func, *, target=None, pass_configs=None):
        record = self.records[idx]
        try:
            with self.stage(idx, "elaborate"):
                program = elaborate_func(**dict(config_arg))
        except Exception as error:
            record.update(status="elaboration_failed", error=str(error))
            raise
        with self.stage(idx, "analysis"):
            try:
                result = analyze_prim_func(
                    program,
                    self.config,
                    target=target if target is not None else self.target,
                    device_limits=self.device_limits,
                    pass_configs=pass_configs,
                )
                record.update(result)
                record["pre_lowering"] = result["pressure"]["decision"]
            except Exception as error:
                # Analysis is not compilation: a failure supplies no rejection evidence.
                record["analysis_error"] = str(error)
                record["pre_lowering"] = {"keep": True, "would_reject": False, "status": "unknown"}
        if not record["pre_lowering"]["keep"]:
            record["status"] = "pre_lowering_rejected"
            reasons = "; ".join(record["pre_lowering"].get("reasons", [])) or "register pressure exceeds the configured limit"
            raise CarverReject(f"config {idx}: {reasons}")
        record["status"] = "analyzed"
        return program

    def post_compile(self, idx, resource_usage, launch_infos, *, target=None):
        record = self.records[idx]
        decision = check_compiler_resources(
            resource_usage,
            [info.function_name for info in launch_infos],
            self.config,
            target=target if target is not None else self.target,
            launch_infos=launch_infos,
            device_limits=self.device_limits,
        )
        record["post_compile"] = decision
        record["compiler_resources"] = decision["resources"]
        if not decision["keep"]:
            record["status"] = "post_compile_rejected"
            raise CarverReject("; ".join(decision["reasons"]))

    def compilation_result(self, idx, error):
        record = self.records[idx]
        if error is None:
            record["status"] = "compiled"
        elif record["status"] not in ("elaboration_failed", "pre_lowering_rejected", "post_compile_rejected"):
            record.update(status="compilation_failed", error=str(error))

    def benchmark_result(self, idx, status, latency, error):
        self.records[idx].update(status="benchmarked" if status == "ok" else f"benchmark_{status}", latency_ms=latency, error=error)

    def finish(self):
        totals = {}
        for record in self.records:
            for stage, duration in record["timings_ms"].items():
                totals[stage] = totals.get(stage, 0) + duration
        cost = sum(totals.values())
        ranking = rank_records(self.records) if self.config.ranking else []
        for entry in ranking:
            self.records[entry["index"]]["ranking"] = entry
        report = {
            "analysis_version": ANALYSIS_VERSION,
            "settings": self.config.to_cache_key_dict(),
            "device_limits": self.device_limits,
            "ranking": ranking,
            "ranking_note": "All configs retained in the report; scores use only pre-lowering tile analysis. No top-K cutoff is applied.",
            "configs": self.records,
            "wall_time_ms": (time.perf_counter() - self.started) * 1000,
            "stage_cost_ms": totals,
            "stage_cost_percent": {k: v / cost * 100 if cost else 0 for k, v in totals.items()},
            "timing_note": "Stage costs are summed work, including proportional shares of grouped compilation; wall time includes parallel overlap and benchmarking.",
        }
        if self.config.report_path:
            path = Path(self.config.report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return report

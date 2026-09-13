"""Benchmark a frozen XGBoost selection using the ordinary autotuner."""

from collections import Counter
import csv
import time

from experiments._common import observe_compilation


def prepare_selected(builder, configs, selected, target):
    """An invalid first selection must not prevent trying the rest of fixed K.

    The ordinary autotuner validates its first PrimFunc before worker launch.
    Resolve selected elaboration failures here, without adding replacements.
    """
    from tvm.target import Target

    valid, failures = [], {}
    with Target(target):
        for index in selected:
            kwargs = {key: value for key, value in configs[index].items() if key != "pass_configs"}
            try:
                builder(**kwargs)
            except Exception as error:
                failures[index] = dict(status="elaboration_failed", error=str(error))
            else:
                valid.append(index)
    return valid, failures


def run_selected(case, configs, original_indices, target, inputs, expected, settings, output, report):
    from tilelang.autotuner import AutoTuner
    from experiments.portable.run import write_json

    selected = report["selection"]["selected_indices"]
    outcomes = {}

    class ObservedTuner(AutoTuner):
        def _prepare_compile_execution(self, *args, **kwargs):
            return observe_compilation(super()._prepare_compile_execution(*args, **kwargs), outcomes)

    for record in report["configs"]:
        record["original_index"] = original_indices[record["index"]]
    write_json(output / "xgboost.json", report)  # Freeze before any compilation.
    started = time.perf_counter()
    supplied, failures = prepare_selected(case.build, configs, selected, target)
    for index, failure in failures.items():
        report["configs"][index].update(failure)
    tuner = (
        (
            ObservedTuner(case.build, [configs[index] for index in supplied])
            .set_compile_args(target=target, execution_backend="tvm_ffi", out_idx=case.out_idx, pass_configs=case.pass_configs)
            .set_profile_args(
                supply_prog=lambda params: inputs,
                ref_prog=lambda *args: expected,
                manual_check_prog=case.check,
                backend="event",
                rtol=case.rtol,
                atol=case.atol,
                max_mismatched_ratio=0.0,
            )
            .set_benchmark_report_path(str(output / "benchmarks.tsv"))
        )
        if supplied
        else None
    )
    error, winner = None, None
    try:
        if not supplied:
            raise RuntimeError("Auto-tuning failed: all selected candidates failed elaboration")
        winner = tuner.run(warmup=settings["warmup"], rep=settings["rep"], timeout=settings["timeout"], early_stop=False)
    except RuntimeError as failure:
        if not str(failure).startswith("Auto-tuning failed:"):
            raise
        error = str(failure)
    duration = time.perf_counter() - started
    if (output / "benchmarks.tsv").exists():
        with (output / "benchmarks.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                outcomes[int(row["index"])] = dict(
                    status="benchmarked" if row["status"] == "ok" else "benchmark_" + row["status"],
                    latency_ms=float(row["latency_ms"]) if row["latency_ms"] else None,
                    error=row["error"] or None,
                )
    for local, index in enumerate(supplied):
        report["configs"][index].update(outcomes.get(local, dict(status="not_attempted")))
    write_json(output / "xgboost.json", report)
    write_json(output / "outcomes.json", report["configs"])
    result = dict(
        status="completed" if winner is not None else "failed",
        reason=error,
        metric=report["metric"],
        score_units=report["score_units"],
        model_sha256=report["model_sha256"],
        selection=report["selection"],
        configs=len(configs),
        selection_seconds=report["selection"]["wall_time_ms"] / 1000,
        compile_and_benchmark_seconds=duration,
        tuning_seconds=duration + report["selection"]["wall_time_ms"] / 1000,
        candidate_statuses=dict(Counter(record["status"] for record in report["configs"])),
    )
    if winner is not None:
        index = configs.index(winner.config)
        ranked = next(row for row in report["ranking"] if row["index"] == index)
        result.update(
            correctness="passed",
            winner=dict(
                index=index,
                original_index=original_indices[index],
                config=winner.config,
                latency_ms=winner.latency,
                predicted_rank=ranked["rank"],
            ),
        )
    return result

"""Benchmark a frozen common-grid selection using the ordinary autotuner."""

from collections import Counter
import csv
import hashlib
import time

from experiments.utils.cli import observe_compilation


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


def run_selected(case, configs, original_indices, target, inputs, expected, settings, output, report, *, report_name="xgboost"):
    from tilelang.autotuner import AutoTuner
    from experiments.utils.io import write_json

    selected = report["selection"]["selected_indices"]
    outcomes = {}

    class ObservedTuner(AutoTuner):
        def _prepare_compile_execution(self, *args, **kwargs):
            execution = observe_compilation(super()._prepare_compile_execution(*args, **kwargs), outcomes)
            for future in execution[2]:

                def record_source(done):
                    try:
                        results = done.result()
                    except Exception:
                        return
                    for index, _, kernel, error in results:
                        if not error and kernel is not None:
                            source = kernel.get_kernel_source()
                            outcomes.setdefault(index, {}).update(program_sha256=hashlib.sha256(source.encode()).hexdigest())

                future.add_done_callback(record_source)
            return execution

    for record in report["configs"]:
        from .spaces import config_id

        record["original_index"] = original_indices[record["index"]]
        record["config_id"] = config_id(record["config"])
    write_json(output / f"{report_name}.json", report)  # Freeze before any compilation.
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
    empty_carver = report_name == "carver" and not selected
    try:
        if empty_carver:
            error = "Carver's unchanged policy rejected every supplied configuration; no fallback selection"
        elif not supplied:
            raise RuntimeError("Auto-tuning failed: all selected candidates failed elaboration")
        else:
            winner = tuner.run(warmup=settings["warmup"], rep=settings["rep"], timeout=settings["timeout"], early_stop=False)
    except RuntimeError as failure:
        if not str(failure).startswith("Auto-tuning failed:"):
            raise
        error = str(failure)
    duration = time.perf_counter() - started
    if (output / "benchmarks.tsv").exists():
        with (output / "benchmarks.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                outcomes.setdefault(int(row["index"]), {}).update(
                    status="benchmarked" if row["status"] == "ok" else "benchmark_" + row["status"],
                    latency_ms=float(row["latency_ms"]) if row["latency_ms"] else None,
                    error=row["error"] or None,
                )
    for local, index in enumerate(supplied):
        report["configs"][index].update(outcomes.get(local, dict(status="not_attempted")))
    write_json(output / f"{report_name}.json", report)
    write_json(output / "outcomes.json", report["configs"])
    write_json(output / "compilation-census.json", compilation_census(report["configs"]))
    result = dict(
        status="completed" if winner is not None else "model_unavailable" if empty_carver else "failed",
        reason=error,
        metric=report["metric"],
        score_units=report["score_units"],
        model_sha256=report.get("model_sha256"),
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
                predicted_rank=ranked["rank"] if ranked.get("score") is not None else None,
            ),
        )
    return result


def compilation_census(records):
    """Generated device source identity is conservative, not binary equivalence."""
    compiled = [r for r in records if r.get("program_sha256")]
    correct = [r for r in records if r["status"] == "benchmarked"]
    return dict(
        candidate_count=len(records),
        compiled_count=len(compiled),
        correct_count=len(correct),
        distinct_program_count=len({r["program_sha256"] for r in compiled}),
        distinct_correct_program_count=len({r["program_sha256"] for r in correct if r.get("program_sha256")}),
        identity="sha256 of generated device source; binary equivalence is not claimed",
        statuses=dict(Counter(r["status"] for r in records)),
    )

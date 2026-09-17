"""Analysis, compilation and correctness smoke checks without candidate timing."""

from collections import Counter
import hashlib
import re
import time

from tiletune_core.budget import AttemptLedger
from experiments.utils.io import write_json


def instruction_evidence(source, target, operation, config):
    """Inspect generated instruction text; a target name alone is not evidence."""
    matrix = operation in ("gemm", "gemm_fp8", "attention", "kda_chunk_o")
    arch = target.get("arch", "").rstrip("af")
    required = []
    if target["kind"] == "cuda":
        if matrix:
            required = (
                ["mma.sync"]
                if arch in ("sm_80", "sm_86", "sm_89")
                else ["wgmma.mma_async"]
                if arch == "sm_90"
                else ["tcgen05.mma", "tcgen05.ld", "tcgen05.commit"]
            )
        if operation == "gemm_fp8":
            required += ["e5m2" if "e5m2" in source else "e4m3"]
        if config.get("stages", config.get("num_stages", 0)):
            required += ["cp.async"] if arch in ("sm_80", "sm_86", "sm_89") else ["cp.async.bulk.tensor", "mbarrier"]
    elif target["kind"] == "hip" and matrix:
        required = ["mfma"]
    found = {pattern: bool(re.search(re.escape(pattern), source, re.IGNORECASE)) for pattern in required}
    return dict(status="verified" if required and all(found.values()) else "missing" if required else "not_applicable", required=found)


def run_smoke(case, configs, original_indices, target, inputs, expected, settings, output, report):
    import tilelang

    ledger = AttemptLedger(len(configs), list(range(len(configs))))
    expected = expected if isinstance(expected, tuple | list) else [expected]
    started = time.perf_counter()
    records = report["configs"]
    times = dict(compilation=0.0, correctness=0.0, instruction_inspection=0.0)
    for i, config in enumerate(configs):
        ledger.start(i)
        record = records[i]
        record["original_index"] = original_indices[i]
        try:
            kwargs = {k: v for k, v in config.items() if k != "pass_configs"}
            t = time.perf_counter()
            kernel = tilelang.compile(
                case.build(**kwargs),
                target=target,
                execution_backend="tvm_ffi",
                out_idx=case.out_idx,
                pass_configs={**case.pass_configs, **config.get("pass_configs", {})},
            )
            times["compilation"] += time.perf_counter() - t
            source = kernel.get_kernel_source()
            record["program_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            record["status"] = "compiled"
            t = time.perf_counter()
            result = kernel(*inputs)
            case.check(result if isinstance(result, tuple | list) else [result], expected)
            times["correctness"] += time.perf_counter() - t
            record["status"] = "correct"
            t = time.perf_counter()
            if target["kind"] == "cuda":
                source = kernel._get_ptx()
                (output / f"candidate-{i}.ptx").write_text(source)
                record["instruction_evidence"] = instruction_evidence(source, target, settings["operation"], config)
                record["instruction_evidence"]["basis"] = "generated PTX"
            else:
                record["instruction_evidence"] = dict(status="unverified", reason="native ISA extraction required")
            times["instruction_inspection"] += time.perf_counter() - t
            ledger.finish(i, "passed")
        except Exception as error:
            record["status"] = (
                "correctness_failed"
                if record.get("status") == "compiled"
                else "instruction_inspection_failed"
                if record.get("status") == "correct"
                else "compilation_failed"
            )
            record["error"] = f"{type(error).__name__}: {error}"
            ledger.finish(i, "failed")
        write_json(output / "attempts.json", ledger.to_dict())
        write_json(output / "outcomes.json", records)
    good = [r for r in records if r.get("status") == "correct"]
    evidence = all(r.get("instruction_evidence", {}).get("status") in ("verified", "not_applicable") for r in good)
    analyses = all(r.get("pressure") is not None for r in records)
    write_json(output / "tiletune.json", report)
    return dict(
        status="smoke_passed" if good and evidence and analyses else "failed",
        reason=None
        if good and evidence and analyses
        else "smoke requires completed analysis, correct candidates, and native instruction evidence",
        configs=len(configs),
        correct_candidates=len(good),
        candidate_statuses=dict(Counter(r["status"] for r in records)),
        budget=ledger.to_dict(),
        costs_seconds=times,
        wall_seconds=time.perf_counter() - started,
        measurement=dict(mode="correctness_only", timed_candidates=0),
    )

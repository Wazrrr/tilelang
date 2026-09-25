"""The portable evaluator is testable without a compiler or accelerator."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from tiletune_core import AnalysisReport, KernelFacts, backend_model, classify_bound, evaluate
from tiletune_core.budget import AttemptLedger
from tiletune_core.ranking import rank_records, select_top_k


def fixture(backend="ascend910b"):
    model = backend_model(backend)
    copy, compute = ("mte2", "cube") if backend == "ascend910b" else ("vmem", "mfma")
    scope = "l1" if backend == "ascend910b" else "lds"
    facts = KernelFacts(
        backend,
        model.target,
        operations=[
            dict(id=0, engine=copy, service="copy", work=64, completion_latency="copy_ready", dependencies=[]),
            dict(id=1, engine=compute, service="gemm", work=128, dependencies=[dict(operation=0, distance=0)]),
        ],
        regions=[dict(operations=[0, 1], iterations=1)],
        storage=[dict(scope=scope, peak=17)],
        launch=dict(tasks=1),
    )
    model = replace(
        model,
        rates=dict(copy=16, gemm=32),
        latencies=dict(copy_ready=3),
        capacities={scope: 64},
        allocation_units={scope: 16},
        units=1,
        max_resident=1,
    )
    return facts, model


def test_bound_classifier_is_exported():
    assert classify_bound(1024, 4, 200) == "compute"
    assert classify_bound(100, 4, 200) == "memory"
    assert classify_bound(None, 4, 200) is None


@pytest.mark.parametrize("backend", ["ascend910b", "cdna4"])
def test_async_completion_and_native_allocations(backend):
    facts, model = fixture(backend)
    report = evaluate(facts, model)
    assert report.score == 11  # copy service 4, completion 3, dependent matrix service 4
    assert list(report.resources["allocations"].values()) == [32]
    assert AnalysisReport.from_dict(report.to_dict()) == report
    assert KernelFacts.from_dict(facts.to_dict()) == facts


def test_compressed_repetition_and_buffer_reuse():
    facts, model = fixture()
    facts.operations[0]["dependencies"] = [dict(operation=1, distance=1)]
    facts.regions[0]["iterations"] = 1_000_000
    assert evaluate(facts, model).score == 11_000_000


def test_independent_engines_overlap_and_zero_work_region_is_legal():
    facts, model = fixture()
    facts.operations[1]["dependencies"] = []
    assert evaluate(facts, model).score == 7
    facts.regions[0]["iterations"] = 0
    assert evaluate(facts, model).score == 0


def test_missing_profile_is_unknown_not_zero_or_resource_rejection():
    facts, model = fixture()
    report = evaluate(facts, replace(model, rates={}))
    assert report.score is None
    assert report.diagnostics[0].code == "missing_profile"
    report = evaluate(facts, replace(model, capacities={"l1": 16}))
    assert report.score is None
    assert report.diagnostics[0].category == "resource"


def test_native_targets_do_not_inherit_cuda_or_cdna3():
    facts, model = fixture("cdna4")
    assert model.target["mcpu"] == "gfx950"
    with pytest.raises(ValueError, match="native scopes"):
        replace(model, capacities={"shared": 65536})
    with pytest.raises(ValueError, match="do not match"):
        evaluate(replace(facts, target={"kind": "hip", "mcpu": "gfx942"}), model)
    assert backend_model("blackwell").engines[0] == "tcgen05"
    assert "tmem" in backend_model("blackwell").scopes


def test_strict_facts_reject_compiler_objects_and_nonfinite_work():
    facts, _ = fixture()
    with pytest.raises(ValueError, match="finite JSON"):
        replace(facts, ownership=[dict(layout=object())])
    with pytest.raises(ValueError, match="finite JSON"):
        replace(facts, operations=[dict(work=float("nan"))])
    with pytest.raises(ValueError, match="version"):
        KernelFacts.from_dict(dict(facts.to_dict(), version=99))


def test_failures_consume_attempts_without_replacement_on_resume():
    ledger = AttemptLedger(3, [7, 1])
    ledger.start(7)
    ledger.finish(7, "failed")
    ledger = AttemptLedger.from_dict(ledger.to_dict())
    assert ledger.to_dict()["consumed"] == 1
    assert ledger.to_dict()["unused"] == 1
    with pytest.raises(ValueError, match="previously"):
        ledger.start(7)
    with pytest.raises(ValueError, match="selected"):
        ledger.start(2)
    ledger.start(1)
    ledger.finish(1, "interrupted")
    assert ledger.to_dict()["remaining"] == 0


def test_import_and_evaluation_without_site_packages():
    root = str(Path(__file__).resolve().parents[3])
    code = f"""import sys
sys.path.insert(0, {root!r})
from tiletune_core import KernelFacts, backend_model, evaluate
m = backend_model('ascend910b')
r = evaluate(KernelFacts('ascend910b', m.target), m)
assert r.score is None
assert not any(k.split('.')[0] in ('tilelang', 'torch', 'tvm', 'numpy') for k in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)


def test_nonfinite_scores_form_a_conservative_unknown_tail():
    records = [dict(index=i, tile_cost=dict(score=score)) for i, score in enumerate((2, float("nan"), float("inf"), 2))]
    ranking = rank_records(records)
    assert select_top_k(ranking, 20) == [0, 3, 1, 2]
    assert [r["tier"] for r in ranking] == ["eligible", "eligible", "unknown", "unknown"]
    assert ranking[0]["tie_last_rank"] == 2


def test_resolved_ascend_import_requires_pin_and_verified_events():
    import json
    from experiments.ascend.facts import import_facts

    root = Path(__file__).resolve().parents[3]
    revision = json.loads((root / "experiments/ascend/environment.json").read_text())["revision"]
    facts, model = fixture()
    record = dict(
        compiler_revision=revision,
        source_sha256="test",
        operations=[
            dict(
                index=op["id"],
                engine=op["engine"],
                service=op["service"],
                work=op["work"],
                events=op["dependencies"],
                completion_latency=op.get("completion_latency"),
            )
            for op in facts.operations
        ],
        regions=facts.regions,
        storage=facts.storage,
        launch=facts.launch,
        ownership_verified=True,
        events_verified=True,
    )
    assert evaluate(import_facts(record), model).score == 11
    record["events_verified"] = False
    assert evaluate(import_facts(record), model).score is None
    record["compiler_revision"] = "another-revision"
    with pytest.raises(ValueError, match="pinned compiler"):
        import_facts(record)

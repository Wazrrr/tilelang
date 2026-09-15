"""Stable coverage categories alongside the existing human-readable reasons."""


def analysis_diagnostics(modules):
    entries = []
    pressure = modules["register_pressure"]
    decision = pressure["decision"]
    for field, code in (
        ("physical_reasons", "resource_violation"),
        ("policy_reasons", "policy_rejection"),
        ("uncertainty_reasons", "allocation_uncertainty"),
    ):
        entries.extend(dict(code=code, module="register_pressure", reason=reason) for reason in decision.get(field, []))
    pipeline = modules["pipeline_overlap"]
    entries.extend(dict(module="pipeline_overlap", **item) for item in pipeline.get("diagnostics", []))
    # Legacy reports may have no codes. Preserve their reasons without guessing
    # a failure category from human-readable text.
    if "diagnostics" not in pipeline:
        entries.extend(dict(code="legacy_unknown", module="pipeline_overlap", reason=reason) for reason in pipeline.get("unknown", []))
    for reason in modules["waves"].get("unknown", []):
        entries.append(dict(code="allocation_uncertainty", module="waves", reason=reason))
    return entries

"""Import resolved CDNA4 MFMA/LDS schedules, without CDNA3 capacity defaults."""

from tiletune_core import Diagnostic, KernelFacts, backend_model


def import_cdna4_facts(record):
    model = backend_model("cdna4")
    if record.get("mcpu") != "gfx950":
        raise ValueError("CDNA4 facts require gfx950")
    unknown = [Diagnostic(**item) for item in record.get("unresolved", [])]
    for field, code, reason in (
        ("ownership_verified", "unresolved_ownership", "native MFMA/wave ownership is not verified"),
        ("allocation_verified", "unresolved_allocation", "CU/SIMD register allocation is not verified"),
        ("synchronization_verified", "unresolved_synchronization", "LDS/VMEM wait dependencies are not verified"),
    ):
        if not record.get(field):
            unknown.append(Diagnostic(code, "uncertainty", reason))
    return KernelFacts(
        model.name,
        model.target,
        operations=record["operations"],
        regions=record["regions"],
        ownership=record.get("ownership", []),
        storage=record["storage"],
        launch=record["launch"],
        unresolved=unknown,
        provenance=record["provenance"],
    )

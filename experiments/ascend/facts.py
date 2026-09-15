"""Import resolved Ascend C lowering records into the portable fact contract.

This module does not import either TileLang installation. The native worker must
produce the resolved lowering record after storage planning and event insertion.
Unresolved native ownership or event dependencies remain explicit unknowns.
"""

from pathlib import Path
import json

from tiletune_core import Diagnostic, KernelFacts, backend_model


def import_facts(record):
    pin = json.loads(Path(__file__).with_name("environment.json").read_text())
    if record.get("compiler_revision") != pin["revision"]:
        raise ValueError("Ascend fact producer does not match the pinned compiler")
    model = backend_model("ascend910b")
    operations = []
    for op in record["operations"]:
        operations.append(
            dict(
                id=op["index"],
                engine=op["engine"],
                service=op["service"],
                work=op["work"],
                completion_latency=op.get("completion_latency"),
                dependencies=op["events"],
            )
        )
    unresolved = [Diagnostic(**item) for item in record.get("unresolved", [])]
    if not record.get("ownership_verified"):
        unresolved.append(Diagnostic("unresolved_ownership", "uncertainty", "Ascend Cube/Vector ownership is not verified"))
    if not record.get("events_verified"):
        unresolved.append(Diagnostic("unresolved_synchronization", "uncertainty", "Ascend event insertion is not verified"))
    return KernelFacts(
        model.name,
        model.target,
        operations=operations,
        regions=record["regions"],
        ownership=record.get("ownership", []),
        storage=record["storage"],
        launch=record["launch"],
        unresolved=unresolved,
        provenance=dict(compiler_revision=pin["revision"], source_sha256=record["source_sha256"]),
    )

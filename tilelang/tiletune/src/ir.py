"""Captured tile operations and regions, retaining original IR identities."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Region:
    buffer: object
    ranges: list
    precision: str = "exact"

    @classmethod
    def from_ir(cls, region, precision="exact"):
        return cls(region.buffer, list(region.region), precision)

    def to_dict(self):
        return {
            "buffer": self.buffer.name,
            "buffer_id": str(hash(self.buffer)),
            "scope": self.buffer.scope(),
            "ranges": [{"min": str(r.min), "extent": str(r.extent)} for r in self.ranges],
            "precision": self.precision,
        }


@dataclass
class Operation:
    index: int
    kind: str
    reads: list[Region]
    writes: list[Region]
    metadata: object = None
    loops: tuple = ()
    predicates: tuple = ()
    branches: tuple = ()
    dependencies: list[int] = field(default_factory=list)
    unknown: bool = False
    demands: list[Region] = field(default_factory=list)
    launch_threads: dict = field(default_factory=dict)
    pipeline_stages: tuple = ()

    def to_dict(self):
        return {
            "index": self.index,
            "kind": self.kind,
            "reads": [r.to_dict() for r in self.reads],
            "writes": [r.to_dict() for r in self.writes],
            "dependencies": self.dependencies,
            "loops": [{"var": str(v), "min": str(r.min), "extent": str(r.extent), "kind": k} for v, r, k in self.loops],
            "predicates": [str(p) for p in self.predicates],
            "unknown": self.unknown,
        }


@dataclass
class PropagationResult:
    """One tile-demand graph and its derived per-CTA loop coverage."""

    operations: list[Operation]
    per_iteration_inputs: list[Region]
    full_loop_inputs: list[Region]
    unknown: list[str]
    input_loops: list = field(default_factory=list)

    def to_dict(self):
        return {
            "operations": [op.to_dict() for op in self.operations],
            "per_iteration_inputs": [r.to_dict() for r in self.per_iteration_inputs],
            "full_loop_inputs": [r.to_dict() for r in self.full_loop_inputs],
            "unknown": self.unknown,
        }

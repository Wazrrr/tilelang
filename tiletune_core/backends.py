"""Explicit backend contracts. All capacities and service rates are supplied.

These contracts are not calibrated hardware profiles. In particular CDNA4 never
falls back to CDNA3 limits, and Ascend uses storage/engine names from Ascend C.
"""

from dataclasses import asdict, dataclass, field
from math import isfinite


@dataclass(frozen=True)
class BackendModel:
    name: str
    target: dict
    engines: tuple[str, ...]
    scopes: tuple[str, ...]
    rates: dict = field(default_factory=dict)
    latencies: dict = field(default_factory=dict)
    capacities: dict = field(default_factory=dict)
    allocation_units: dict = field(default_factory=dict)
    units: int | None = None
    max_resident: int | None = None
    profile_identity: dict = field(default_factory=dict)

    def __post_init__(self):
        for values, positive in ((self.rates, True), (self.latencies, False)):
            if any(type(v) not in (int, float) or not isfinite(v) or v < 0 or (positive and not v) for v in values.values()):
                raise ValueError("backend rates must be positive and latencies nonnegative")
        for values in (self.capacities, self.allocation_units):
            if set(values) - set(self.scopes) or any(type(v) is not int or v <= 0 for v in values.values()):
                raise ValueError("storage capacity/allocation must use native scopes and positive integers")
        for value in (self.units, self.max_resident):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("execution-unit count/residency must be positive integers")

    def to_dict(self):
        from .contracts import json_value

        return json_value(dict(version=1, **asdict(self)))

    @classmethod
    def from_dict(cls, value):
        from .contracts import json_value

        value = json_value(value)
        if value.pop("version", None) != 1:
            raise ValueError("unsupported backend profile version")
        value["engines"], value["scopes"] = tuple(value["engines"]), tuple(value["scopes"])
        return cls(**value)


_BACKENDS = {}


def register_backend(name, factory):
    if name in _BACKENDS:
        raise ValueError(f"backend already registered: {name}")
    _BACKENDS[name] = factory


def backend_model(name, **profile):
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unregistered backend: {name}") from None
    return factory(**profile)


def _register(name, target, engines, scopes):
    def factory(**profile):
        return BackendModel(name, dict(target), engines, scopes, **profile)

    register_backend(name, factory)


_register("ampere", {"kind": "cuda", "arch": "sm_80"}, ("mma", "simt", "copy", "lds"), ("register", "shared", "threads", "warps"))
_register("hopper", {"kind": "cuda", "arch": "sm_90a"}, ("wgmma", "simt", "tma", "lds"), ("register", "shared", "threads", "warps"))
_register(
    "blackwell", {"kind": "cuda", "arch": "sm_100a"}, ("tcgen05", "simt", "tma", "tmem"), ("register", "shared", "tmem", "threads", "warps")
)
_register("cdna4", {"kind": "hip", "mcpu": "gfx950"}, ("mfma", "valu", "vmem", "lds"), ("vgpr", "agpr", "sgpr", "lds", "waves"))
_register(
    "ascend910b",
    {"kind": "ascendc", "arch": "Ascend910B"},
    ("cube", "vector", "mte1", "mte2", "mte3"),
    ("l0a", "l0b", "l0c", "l1", "ub", "events"),
)

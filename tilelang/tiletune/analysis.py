"""Public whole-kernel analysis and explicit tile-dependency queries."""

from tvm import tirx as tir
from .config import TileTuneConfig
from .src.ir import Region
from .src.collector import _Collector
from .src.propagation import _propagate_tiles
from .src.ir_utils import resolve_pass_configs


def propagate_inputs(func, outputs):
    """Query explicit, nonempty BufferRegion (or Region) output tiles.

    Both APIs use the same tile propagation. This query does not estimate
    pressure or timing; use analyze_prim_func for whole-kernel resource analysis.
    Launch offsets stay symbolic; full_loop_inputs covers loops within a CTA.
    """
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("propagate_inputs expects an elaborated PrimFunc")
    if outputs is None:
        raise TypeError("propagate_inputs requires explicit output regions")
    regions = [r if isinstance(r, Region) else Region.from_ir(r) for r in outputs]
    if not regions:
        raise ValueError("propagate_inputs requires at least one output region")
    return _propagate_tiles(_Collector(func), regions)


def analyze_prim_func(func, config=None, *, target=None, device_limits=None, pass_configs=None, trace_context=None):
    """Analyze all captured global outputs; invalid inputs and stage errors raise."""
    if not isinstance(func, tir.PrimFunc):
        raise TypeError("analyze_prim_func expects an elaborated PrimFunc")
    config = TileTuneConfig.from_value(config)
    if target is None and func.attrs is not None:
        target = func.attrs.get("target")
    device_limits = config.device_limits if device_limits is None else device_limits
    pass_configs = resolve_pass_configs(func, pass_configs)
    from .engine import analyze_kernel

    return analyze_kernel(func, config, target, device_limits, pass_configs, trace_context)

"""CTA resource limits and launch waves, independent of kernel family."""

from math import prod


def analyze_waves(col, memory, pressure, device_limits=None):
    from .src.ir_utils import _int

    unknown = []
    model = pressure.get("target_model")
    if model is not None and model["kind"] is not None and not model["block_execution"]:
        unknown.append("target requires a non-SIMT core/storage residency model")

    def product(values):
        values = [_int(v) for v in values]
        return prod(values) if all(v is not None and v >= 0 for v in values) else None

    thread_sets = {tuple(sorted((k, str(v)) for k, v in op.launch_threads.items())) for op in col.operations}
    grid = product(v for k, v in col.threads.items() if k.startswith("blockIdx."))
    threads = product(v for k, v in col.threads.items() if k.startswith("threadIdx."))
    if len(thread_sets) != 1 or not any(k.startswith("threadIdx.") for k in col.threads):
        grid = threads = None
        unknown.append("unresolved or multiple launch domains")
    from tiletune_core.occupancy import analyze_waves as evaluate_waves

    return evaluate_waves(dict(grid=grid, threads=threads, unknown=unknown), memory, pressure, device_limits)

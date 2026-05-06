"""Deterministic Hopper-first config preselector for GEMM autotuning."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any

from ..roller.hint import Hint
from ..roller.rasterization import NoRasterization

logger = logging.getLogger(__name__)

SUPPORTED_SELECTORS: tuple[str, ...] = ("none", "hopper_hybrid_v1")


@dataclass
class SelectorTelemetry:
    """Telemetry emitted by one selector invocation."""

    selector_name: str
    selector_input_count: int
    selector_output_count: int
    selector_pool_k: int | None
    selector_time_ms: float
    selector_fallback_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_name": self.selector_name,
            "selector_input_count": self.selector_input_count,
            "selector_output_count": self.selector_output_count,
            "selector_pool_k": self.selector_pool_k,
            "selector_time_ms": self.selector_time_ms,
            "selector_fallback_used": self.selector_fallback_used,
        }


@dataclass
class _NormalizedCandidate:
    payload: Any
    index: int
    signature: tuple[Any, ...]
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    thread_num: int
    use_tc: bool
    use_async: bool


@dataclass
class _ScoredCandidate:
    candidate: _NormalizedCandidate
    score: float
    tail_penalty: float
    smem_pressure: float
    accum_pressure: float
    thread_preference: float
    k_divisibility: float


def default_selector_pool_k(topk: int) -> int:
    """Recommended candidate pool size for selector pre-ranking."""
    return min(max(8 * max(1, int(topk)), 128), 1024)


def is_hopper_or_newer_cuda_arch(arch: Any) -> bool:
    """Returns True when arch metadata indicates CUDA sm90+."""
    if arch is None:
        return False
    platform = str(getattr(arch, "platform", "")).upper()
    if platform != "CUDA":
        return False
    try:
        return int(getattr(arch, "sm_version", -1)) >= 90
    except Exception:
        return False


def validate_selector_name(selector_name: str) -> None:
    if selector_name not in SUPPORTED_SELECTORS:
        raise ValueError(f"Unsupported config selector '{selector_name}'. Supported values: {SUPPORTED_SELECTORS}")


def should_activate_selector(selector_name: str, arch: Any, *, gemm_like: bool) -> bool:
    """Activation policy: only Hopper+ CUDA and GEMM-like workloads."""
    return selector_name == "hopper_hybrid_v1" and gemm_like and is_hopper_or_newer_cuda_arch(arch)


def select_hints(
    hints: list[Hint] | None,
    *,
    topk: int,
    selector_name: str,
    arch: Any,
    m: int,
    n: int,
    k: int,
    in_dtype: Any,
    accum_dtype: Any,
    selector_pool_k: int | None,
    selector_debug: bool,
    gemm_like: bool = True,
) -> tuple[list[Hint], SelectorTelemetry]:
    """Select top-k hint candidates with optional Hopper hybrid selector."""
    validate_selector_name(selector_name)
    start = time.perf_counter()
    hint_list = list(hints or [])
    input_count = len(hint_list)
    telemetry = SelectorTelemetry(
        selector_name="none",
        selector_input_count=input_count,
        selector_output_count=input_count,
        selector_pool_k=selector_pool_k,
        selector_time_ms=0.0,
        selector_fallback_used=False,
    )

    if selector_name == "none" or not should_activate_selector(selector_name, arch, gemm_like=gemm_like):
        telemetry.selector_time_ms = (time.perf_counter() - start) * 1000.0
        return hint_list, telemetry

    try:
        selected, fallback_used = _apply_hopper_hybrid_v1(
            candidates=hint_list,
            normalize_fn=_normalize_hint,
            topk=topk,
            m=m,
            n=n,
            k=k,
            in_dtype=in_dtype,
            accum_dtype=accum_dtype,
            arch=arch,
            selector_debug=selector_debug,
        )
        telemetry.selector_name = selector_name
        telemetry.selector_output_count = len(selected)
        telemetry.selector_fallback_used = fallback_used
        telemetry.selector_time_ms = (time.perf_counter() - start) * 1000.0
        return selected, telemetry
    except Exception as ex:  # pragma: no cover - defensive guard
        logger.warning(
            "Config selector '%s' failed (%s: %s). Falling back to original hint order.",
            selector_name,
            type(ex).__name__,
            ex,
        )
        fallback = hint_list[: max(0, min(topk, len(hint_list)))]
        telemetry.selector_name = selector_name
        telemetry.selector_output_count = len(fallback)
        telemetry.selector_fallback_used = True
        telemetry.selector_time_ms = (time.perf_counter() - start) * 1000.0
        return fallback, telemetry


def select_configs(
    configs: list[dict[str, Any]] | None,
    *,
    topk: int,
    selector_name: str,
    arch: Any,
    m: int,
    n: int,
    k: int,
    in_dtype: Any,
    accum_dtype: Any,
    selector_pool_k: int | None,
    selector_debug: bool,
    gemm_like: bool = True,
) -> tuple[list[dict[str, Any]], SelectorTelemetry]:
    """Select top-k config dictionaries with optional Hopper hybrid selector."""
    validate_selector_name(selector_name)
    start = time.perf_counter()
    config_list = list(configs or [])
    input_count = len(config_list)
    telemetry = SelectorTelemetry(
        selector_name="none",
        selector_input_count=input_count,
        selector_output_count=input_count,
        selector_pool_k=selector_pool_k,
        selector_time_ms=0.0,
        selector_fallback_used=False,
    )

    if selector_name == "none" or not should_activate_selector(selector_name, arch, gemm_like=gemm_like):
        telemetry.selector_time_ms = (time.perf_counter() - start) * 1000.0
        return config_list, telemetry

    try:
        selected, fallback_used = _apply_hopper_hybrid_v1(
            candidates=config_list,
            normalize_fn=_normalize_config,
            topk=topk,
            m=m,
            n=n,
            k=k,
            in_dtype=in_dtype,
            accum_dtype=accum_dtype,
            arch=arch,
            selector_debug=selector_debug,
        )
        telemetry.selector_name = selector_name
        telemetry.selector_output_count = len(selected)
        telemetry.selector_fallback_used = fallback_used
        telemetry.selector_time_ms = (time.perf_counter() - start) * 1000.0
        return selected, telemetry
    except Exception as ex:  # pragma: no cover - defensive guard
        logger.warning(
            "Config selector '%s' failed (%s: %s). Falling back to original config order.",
            selector_name,
            type(ex).__name__,
            ex,
        )
        fallback = config_list[: max(0, min(topk, len(config_list)))]
        telemetry.selector_name = selector_name
        telemetry.selector_output_count = len(fallback)
        telemetry.selector_fallback_used = True
        telemetry.selector_time_ms = (time.perf_counter() - start) * 1000.0
        return fallback, telemetry


def _apply_hopper_hybrid_v1(
    *,
    candidates: list[Any],
    normalize_fn,
    topk: int,
    m: int,
    n: int,
    k: int,
    in_dtype: Any,
    accum_dtype: Any,
    arch: Any,
    selector_debug: bool,
) -> tuple[list[Any], bool]:
    """Stage A/B/C: filter -> score -> diversity-constrained select."""
    target_count = max(0, min(int(topk), len(candidates)))
    if target_count == 0:
        return [], False

    normalized = _deduplicate_and_filter(candidates=candidates, normalize_fn=normalize_fn)
    if not normalized:
        logger.warning("Config selector produced no valid candidates after Stage A; falling back to original order.")
        return candidates[:target_count], True

    scored = [
        _score_candidate(
            candidate=entry,
            m=m,
            n=n,
            k=k,
            in_dtype=in_dtype,
            accum_dtype=accum_dtype,
            arch=arch,
        )
        for entry in normalized
    ]
    scored.sort(
        key=lambda item: (
            -item.score,
            item.tail_penalty,
            item.smem_pressure,
            item.accum_pressure,
            -item.thread_preference,
            -item.k_divisibility,
            item.candidate.signature,
            item.candidate.index,
        )
    )

    per_bucket_cap = max(1, int(math.floor(target_count * 0.40)))
    bucket_counts: dict[tuple[int, int], int] = {}
    selected: list[_ScoredCandidate] = []
    deferred: list[_ScoredCandidate] = []

    for item in scored:
        bucket = (item.candidate.block_m, item.candidate.block_n)
        current = bucket_counts.get(bucket, 0)
        if current < per_bucket_cap:
            selected.append(item)
            bucket_counts[bucket] = current + 1
            if len(selected) == target_count:
                break
        else:
            deferred.append(item)

    if len(selected) < target_count:
        for item in deferred:
            selected.append(item)
            if len(selected) == target_count:
                break

    if len(selected) < target_count:
        logger.warning(
            "Config selector selected %d/%d candidates. Falling back to original order.",
            len(selected),
            target_count,
        )
        return candidates[:target_count], True

    if selector_debug:
        top_debug = selected[: min(5, len(selected))]
        logger.info(
            "hopper_hybrid_v1 selected %d/%d candidates (diversity cap=%d). top=%s",
            len(selected),
            len(candidates),
            per_bucket_cap,
            [
                {
                    "score": round(item.score, 4),
                    "block": (item.candidate.block_m, item.candidate.block_n, item.candidate.block_k),
                    "thread_num": item.candidate.thread_num,
                    "num_stages": item.candidate.num_stages,
                }
                for item in top_debug
            ],
        )

    return [item.candidate.payload for item in selected], False


def _deduplicate_and_filter(*, candidates: list[Any], normalize_fn) -> list[_NormalizedCandidate]:
    deduped: list[_NormalizedCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for index, candidate in enumerate(candidates):
        normalized = normalize_fn(candidate=candidate, index=index)
        if normalized is None:
            continue
        if normalized.signature in seen:
            continue
        seen.add(normalized.signature)
        deduped.append(normalized)
    return deduped


def _normalize_hint(*, candidate: Hint, index: int) -> _NormalizedCandidate | None:
    block = _to_int_tuple(getattr(candidate, "block", []))
    if len(block) < 2:
        return None
    block_m, block_n = block[-2], block[-1]

    rstep = _to_int_tuple(getattr(candidate, "rstep", []))
    if len(rstep) < 1:
        return None
    block_k = rstep[0]

    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        return None

    pipeline_stage = max(1, _safe_int(getattr(candidate, "pipeline_stage", 1), default=1))
    num_stages = pipeline_stage if pipeline_stage > 1 else 0

    use_tc = bool(getattr(candidate, "use_tc", False))
    use_async = bool(getattr(candidate, "use_async", False))

    warp = _to_int_tuple(getattr(candidate, "warp", []))
    thread_num = -1
    if len(warp) >= 2 and warp[-2] > 0 and warp[-1] > 0 and block_m % warp[-2] == 0 and block_n % warp[-1] == 0:
        thread_num = (block_m // warp[-2]) * (block_n // warp[-1]) * 32
    elif getattr(candidate, "thread", None):
        thread_shape = _to_int_tuple(getattr(candidate, "thread", []))
        if thread_shape:
            thread_num = math.prod(thread_shape)

    if not _is_hardware_legal(block_m=block_m, block_n=block_n, block_k=block_k, thread_num=thread_num, num_stages=num_stages):
        return None

    rasterization_plan = getattr(candidate, "rasterization_plan", None)
    enable_rasterization = not isinstance(rasterization_plan, NoRasterization)

    signature = (
        "hint",
        block_m,
        block_n,
        block_k,
        num_stages,
        thread_num,
        use_tc,
        use_async,
        enable_rasterization,
    )
    return _NormalizedCandidate(
        payload=candidate,
        index=index,
        signature=signature,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        thread_num=thread_num,
        use_tc=use_tc,
        use_async=use_async,
    )


def _normalize_config(*, candidate: dict[str, Any], index: int) -> _NormalizedCandidate | None:
    if not isinstance(candidate, dict):
        return None

    block_m = _safe_int(candidate.get("block_M"), default=-1)
    block_n = _safe_int(candidate.get("block_N"), default=-1)
    block_k = _safe_int(candidate.get("block_K"), default=-1)
    num_stages = max(0, _safe_int(candidate.get("num_stages"), default=-1))
    thread_num = _safe_int(candidate.get("thread_num"), default=-1)
    use_tc = bool(candidate.get("use_tc", True))
    use_async = bool(candidate.get("use_async", num_stages >= 2))

    if not _is_hardware_legal(block_m=block_m, block_n=block_n, block_k=block_k, thread_num=thread_num, num_stages=num_stages):
        return None

    enable_rasterization = bool(candidate.get("enable_rasteration", False))

    signature = (
        "cfg",
        block_m,
        block_n,
        block_k,
        num_stages,
        thread_num,
        use_tc,
        use_async,
        enable_rasterization,
    )
    return _NormalizedCandidate(
        payload=candidate,
        index=index,
        signature=signature,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        thread_num=thread_num,
        use_tc=use_tc,
        use_async=use_async,
    )


def _is_hardware_legal(*, block_m: int, block_n: int, block_k: int, thread_num: int, num_stages: int) -> bool:
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        return False
    if block_m > 2048 or block_n > 2048 or block_k > 1024:
        return False
    if thread_num <= 0 or thread_num > 1024 or thread_num % 32 != 0:
        return False
    if num_stages < 0 or num_stages > 8:
        return False
    return True


def _score_candidate(
    *,
    candidate: _NormalizedCandidate,
    m: int,
    n: int,
    k: int,
    in_dtype: Any,
    accum_dtype: Any,
    arch: Any,
) -> _ScoredCandidate:
    arch_sm = _safe_int(getattr(arch, "sm_version", -1), default=-1)
    in_nbytes = _dtype_nbytes(in_dtype)
    accum_nbytes = _dtype_nbytes(accum_dtype, default=4)
    smem_cap = max(1, _safe_int(getattr(arch, "smem_cap", 228 * 1024), default=228 * 1024))

    wgmma_candidate = _wgmma_candidate_score(candidate=candidate, arch_sm=arch_sm)
    async_copy_hint = 1.0 if candidate.use_async or candidate.num_stages >= 2 else (0.25 if candidate.num_stages == 1 else 0.0)
    tail_penalty = _tail_penalty(m=m, n=n, k=k, block_m=candidate.block_m, block_n=candidate.block_n, block_k=candidate.block_k)

    stage_factor = 1.0 + 0.25 * max(candidate.num_stages - 1, 0)
    smem_bytes = (candidate.block_m * candidate.block_k + candidate.block_n * candidate.block_k) * in_nbytes * stage_factor
    smem_pressure = min(3.0, smem_bytes / smem_cap)

    accum_per_thread = (candidate.block_m * candidate.block_n * accum_nbytes) / max(1, candidate.thread_num)
    accum_pressure = min(3.0, accum_per_thread / 256.0)

    thread_preference = _thread_preference_score(candidate.thread_num)
    k_divisibility = _k_divisibility_score(k=k, block_k=candidate.block_k)

    score = (
        4.0 * wgmma_candidate
        + 1.5 * async_copy_hint
        + 1.0 * thread_preference
        + 0.8 * k_divisibility
        - 1.3 * tail_penalty
        - 1.0 * smem_pressure
        - 0.8 * accum_pressure
    )

    return _ScoredCandidate(
        candidate=candidate,
        score=score,
        tail_penalty=tail_penalty,
        smem_pressure=smem_pressure,
        accum_pressure=accum_pressure,
        thread_preference=thread_preference,
        k_divisibility=k_divisibility,
    )


def _wgmma_candidate_score(*, candidate: _NormalizedCandidate, arch_sm: int) -> float:
    if arch_sm < 90 or not candidate.use_tc:
        return 0.0
    score = 1.0
    if candidate.block_m < 64:
        score -= 0.25
    if candidate.block_n < 64:
        score -= 0.25
    if candidate.block_k < 16:
        score -= 0.25
    if candidate.thread_num not in (128, 256, 512):
        score -= 0.20
    return max(0.0, min(1.0, score))


def _tail_penalty(*, m: int, n: int, k: int, block_m: int, block_n: int, block_k: int) -> float:
    m_tail = (m % block_m) / block_m if block_m > 0 else 1.0
    n_tail = (n % block_n) / block_n if block_n > 0 else 1.0
    k_tail = (k % block_k) / block_k if block_k > 0 else 1.0
    return m_tail + n_tail + 0.5 * k_tail


def _thread_preference_score(thread_num: int) -> float:
    if thread_num == 256:
        return 1.0
    if thread_num == 128:
        return 0.85
    if thread_num == 512:
        return 0.65
    if thread_num == 64:
        return 0.35
    return max(0.0, 1.0 - abs(thread_num - 256) / 512.0)


def _k_divisibility_score(*, k: int, block_k: int) -> float:
    if block_k <= 0:
        return 0.0
    remainder = k % block_k
    if remainder == 0:
        return 1.0
    return max(0.0, 1.0 - remainder / block_k)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, tuple):
        return tuple(_safe_int(v, default=-1) for v in values)
    if isinstance(values, list):
        return tuple(_safe_int(v, default=-1) for v in values)
    return ()


def _dtype_nbytes(dtype: Any, *, default: int = 2) -> int:
    if dtype is None:
        return default
    text = str(dtype).lower()
    if "float64" in text or "int64" in text:
        return 8
    if "float32" in text or "int32" in text:
        return 4
    if "float16" in text or "bfloat16" in text:
        return 2
    if "float8" in text or "e4m3" in text or "e5m2" in text:
        return 1
    if "int8" in text or "uint8" in text:
        return 1
    if "int4" in text or "uint4" in text:
        return 1
    if "int2" in text or "int1" in text:
        return 1
    return default

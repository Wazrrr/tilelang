"""Analyze a standalone softmax PrimFunc without a family adapter or GPU.

python -m experiments.softmax.analyze --output experiments/results/softmax-analysis.json
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import tilelang.language as T
from tilelang.tiletune import analyze_prim_func
from tiletune_core.ranking import alpha_budget, rank_records, select_top_k


def softmax(rows, columns, block_rows, threads):
    """A stable row softmax, including masks for padded rows and columns."""
    block_columns = 1 << (columns - 1).bit_length()

    @T.prim_func
    def kernel(A: T.Tensor((rows, columns), "float16"), B: T.Tensor((rows, columns), "float16")):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as block:
            values = T.alloc_fragment((block_rows, block_columns), "float32")
            row_max = T.alloc_fragment((block_rows,), "float32")
            row_sum = T.alloc_fragment((block_rows,), "float32")
            for i, j in T.Parallel(block_rows, block_columns):
                values[i, j] = T.if_then_else(
                    (block * block_rows + i < rows) & (j < columns),
                    T.cast(A[block * block_rows + i, j], "float32"),
                    -T.infinity("float32"),
                )
            T.reduce_max(values, row_max, dim=1, clear=True)
            for i, j in T.Parallel(block_rows, block_columns):
                values[i, j] = T.exp(values[i, j] - row_max[i])
            T.reduce_sum(values, row_sum, dim=1, clear=True)
            for i, j in T.Parallel(block_rows, block_columns):
                if (block * block_rows + i < rows) & (j < columns):
                    B[block * block_rows + i, j] = values[i, j] / row_sum[i]

    return kernel


def analyze():
    cases = []
    for rows, columns in ((256, 128), (257, 1000), (4096, 4096)):
        records = []
        for block_rows in (1, 2, 4, 8):
            for threads in (128, 256):
                func = softmax(rows, columns, block_rows, threads)
                before = func.script()
                report = analyze_prim_func(
                    func,
                    {"ranking_metric": "memory", "memory_diagnostics": True},
                    target={"kind": "cuda", "arch": "sm_90a"},
                    device_limits={"sm_count": 132},
                )
                assert func.script() == before
                assert report["specialization"]["name"] == "generic"
                assert report["tile_cost"]["score"] is not None
                assert not report["tile_cost"]["unknown"]
                assert not report["pressure"]["decision"]["would_reject"]
                operations = report["tile_propagation"]["operations"]
                kinds = Counter(op["kind"] for op in operations)
                assert kinds["reduce"] == 2
                assert any(op["dependencies"] for op in operations)
                records.append(
                    dict(
                        index=len(records),
                        config=dict(block_rows=block_rows, threads=threads),
                        tile_cost=report["tile_cost"],
                        pre_lowering=report["pressure"]["decision"],
                        operation_kinds=dict(kinds),
                        analysis=report,
                    )
                )
        ranking = rank_records(records)
        budget = alpha_budget(len(records), 0.5)
        selected = select_top_k(ranking, budget, strict_budget=True)
        cases.append(
            dict(
                shape=[rows, columns],
                configs=records,
                ranking=ranking,
                selection=dict(alpha=0.5, requested_k=budget, selected_indices=selected, selected_count=len(selected)),
            )
        )
        pairs = {(r["score"], r["tie_break_score"]) for r in ranking}
        print(
            f"{rows}x{columns}: {len(records)}/{len(records)} scored and eligible; {len(pairs)} score groups; strict 50% keeps {len(selected)}"
        )
    return dict(
        scope="CPU PrimFunc analysis only; no lowering, GPU execution, correctness comparison, or oracle measurement",
        target="cuda sm_90a; 132 SMs supplied as device metadata",
        family_adapter=False,
        compute_profile=False,
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def forbidden(*args, **kwargs):
        raise AssertionError("standalone softmax analysis invoked a family or timing helper")

    disabled = (
        "tilelang.tiletune.engine.select_specialization",
        "tilelang.tiletune.families.base.KernelSpecialization.__init__",
        "tilelang.tiletune.pipeline.analyze_pipeline",
        "tilelang.tiletune.occupancy.analyze_waves",
        "tilelang.tiletune.engine.predict_warp_specialization",
    )
    with ExitStack() as stack:
        for name in disabled:
            stack.enter_context(patch(name, forbidden))
        result = analyze()
    result["disabled_helpers"] = list(disabled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

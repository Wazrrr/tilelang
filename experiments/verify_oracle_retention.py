"""Verify every fresh oracle-optimal config is retained by live TileTune alpha selection.

Joins a brute-force sweep (experiment 1/2) against the frozen ``tiletune.json``
selection written by the live TileTune system run (experiment 3). No compiler or
GPU import is required; this uses saved JSON only.

Example:
    python -m experiments.verify_oracle_retention \
      --oracle-root experiments/results/three-exp/exp2-bruteforce-mg4 \
      --tiletune-root experiments/results/three-exp/exp3-tiletune-alpha50-pipeline-mg4-grouped8 \
      --alpha 0.5 \
      --output experiments/results/three-exp/oracle-retention.json
"""

import argparse
import json
import math
from pathlib import Path

from experiments.families import FAMILIES


def family_for(op):
    return FAMILIES[op]


def load_manifest_workloads():
    return json.loads(Path("experiments/manifests/five_target_final.json").read_text())["workloads"]


def alpha_budget(pool_size, alpha):
    if type(pool_size) is not int or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    budget = math.floor(pool_size * alpha)
    if budget == 0:
        raise ValueError("alpha selects no candidates from the supplied pool")
    return budget


def find_tiletune_report(tiletune_root, family, workload):
    base = Path(tiletune_root) / family / workload
    matches = sorted(base.rglob("tiletune.json"))
    if not matches:
        raise FileNotFoundError(f"no tiletune.json under {base}")
    if len(matches) > 1:
        raise ValueError(f"multiple tiletune.json under {base}: {matches}")
    return matches[0]


def unsupported_summary(tiletune_root, family, workload):
    base = Path(tiletune_root) / family / workload
    for path in sorted(base.rglob("summary.json")):
        data = json.loads(path.read_text())
        if data.get("status") == "unsupported":
            return path, data
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--tiletune-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("--alpha must be finite and in (0, 1]")

    from experiments.common.spec import Device, Workload, support_reason
    from experiments.utils.results import config_key, load_oracle

    oracle_root = args.oracle_root.resolve()
    tiletune_root = args.tiletune_root.resolve()
    rows = []
    for item in load_manifest_workloads():
        workload = Workload(**item)
        family = family_for(workload.op)
        device = Device(name="ampere", target={"kind": "cuda", "arch": "sm_80"})
        reason = support_reason(workload, device)
        if reason:
            path, summary = unsupported_summary(tiletune_root, family, workload.name)
            row = dict(
                workload=workload.name,
                family=family,
                op=workload.op,
                status="unsupported",
                reason=reason,
                retained=None,
                verified_unsupported=summary is not None,
                summary_path=str(path) if path is not None else None,
            )
            rows.append(row)
            continue

        oracle = load_oracle(oracle_root / workload.name)
        best_ms = oracle["winner"]["latency_ms"]
        oracle_optimal = [
            record
            for record in oracle["records"].values()
            if record["status"] == "benchmarked" and record["latency_ms"] == best_ms
        ]
        if not oracle_optimal:
            raise ValueError(f"{workload.name}: no oracle-optimal measured configs")

        report_path = find_tiletune_report(tiletune_root, family, workload.name)
        report = json.loads(report_path.read_text())
        records = {config_key(record["config"]): record for record in report["configs"]}
        selection = report.get("selection") or {}
        pool_size = selection.get("pool_size", len(report["configs"]))
        budget = alpha_budget(pool_size, args.alpha)

        retained, missing = [], []
        for measured in oracle_optimal:
            record = records.get(config_key(measured["config"]))
            ranking = (record or {}).get("ranking") or {}
            ok = bool(record and record.get("selected") and (ranking.get("tie_last_rank") or math.inf) <= budget)
            entry = dict(
                oracle_index=measured["index"],
                config=measured["config"],
                selected=bool(record and record.get("selected")),
                tie_last_rank=ranking.get("tie_last_rank"),
                retained=ok,
            )
            if ok:
                retained.append(entry)
            else:
                missing.append(entry)

        rows.append(
            dict(
                workload=workload.name,
                family=family,
                op=workload.op,
                status="checked",
                oracle_ms=best_ms,
                oracle_optimal_count=len(oracle_optimal),
                retained_count=len(retained),
                alpha=args.alpha,
                alpha_budget=budget,
                pool_size=pool_size,
                report_path=str(report_path),
                retained=retained,
                missing=missing,
                hits=len(retained) == len(oracle_optimal),
            )
        )

    checked = [r for r in rows if r["status"] == "checked"]
    unsupported = [r for r in rows if r["status"] == "unsupported"]
    result = dict(
        alpha=args.alpha,
        oracle_root=str(oracle_root),
        tiletune_root=str(tiletune_root),
        rows=rows,
        checked_count=len(checked),
        unsupported_count=len(unsupported),
        hits=sum(r["hits"] for r in checked),
        unsupported_verified=all(r["verified_unsupported"] for r in unsupported),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    ok = result["hits"] == len(checked) and result["unsupported_verified"]
    print(
        f"Oracle retention: {result['hits']}/{len(checked)} supported cases retained "
        f"({result['unsupported_count']} unsupported, verified={result['unsupported_verified']})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

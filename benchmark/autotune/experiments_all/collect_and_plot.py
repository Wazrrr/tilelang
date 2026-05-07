#!/usr/bin/env python3
"""Collect experiment CSVs and render per-feature bar charts with matplotlib.

Inputs:
- benchmark/autotune/experiments_grouped_compilation/results/*.csv
- benchmark/autotune/experiments_pipeline_only/results/*.csv
- benchmark/autotune/experiments_multi_gpu_benchmark/results/*.csv
- benchmark/autotune/experiments_all/results/*.csv

Outputs:
- benchmark/autotune/experiments_all/plot_results/combined_summary.csv
- benchmark/autotune/experiments_all/plot_results/combined_summary.md
- benchmark/autotune/experiments_all/plot_results/figures/*.png
- benchmark/autotune/experiments_all/plot_results/figures/*.svg
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
PLOT_ROOT = ROOT / "benchmark" / "autotune" / "experiments_all" / "plot_results"
FIGURE_ROOT = PLOT_ROOT / "figures"

METRICS = ("end_to_end_s", "compilation_s", "benchmark_s", "best_latency_ms")


@dataclass(frozen=True)
class Case:
    suite: str
    label: str
    csv_path: Path
    is_baseline: bool = False


STATIC_CASES: list[Case] = [
    Case(
        suite="grouped_compilation",
        label="G1",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_grouped_compilation" / "results" / "auto_group1.csv",
        is_baseline=True,
    ),
    Case(
        suite="grouped_compilation",
        label="G2",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_grouped_compilation" / "results" / "auto_group2.csv",
    ),
    Case(
        suite="grouped_compilation",
        label="G4",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_grouped_compilation" / "results" / "auto_group4.csv",
    ),
    Case(
        suite="pipeline_only",
        label="NoPipe",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_pipeline_only" / "results" / "auto_nopipeline.csv",
        is_baseline=True,
    ),
    Case(
        suite="pipeline_only",
        label="Pipe",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_pipeline_only" / "results" / "auto_pipeline.csv",
    ),
    Case(
        suite="multi_gpu_benchmark",
        label="1GPU",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_multi_gpu_benchmark" / "results" / "auto_baseline_single_gpu.csv",
        is_baseline=True,
    ),
    Case(
        suite="multi_gpu_benchmark",
        label="MultiGPU",
        csv_path=ROOT / "benchmark" / "autotune" / "experiments_multi_gpu_benchmark" / "results" / "auto_benchmark_multi_gpu.csv",
    ),
]

ALL_FEATURE_INPUT = ROOT / "benchmark" / "autotune" / "experiments_all" / "results"

SHORT_LABEL_MAP = {
    "auto_baseline": "Baseline",
    "auto_group4_pipeline_multigpu": "AllFeat",
}

SUITE_TITLES = {
    "grouped_compilation": "Grouped Compilation",
    "pipeline_only": "Pipeline Only",
    "multi_gpu_benchmark": "Multi-GPU Benchmark",
    "all_features": "All Features Combined",
}

METRIC_TITLE = {
    "end_to_end_s": "End-to-End Time",
    "compilation_s": "Compilation Time",
    "benchmark_s": "Benchmark Time",
    "best_latency_ms": "Best Kernel Latency",
}

METRIC_UNIT = {
    "end_to_end_s": "seconds",
    "compilation_s": "seconds",
    "benchmark_s": "seconds",
    "best_latency_ms": "milliseconds",
}


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def _short_label_from_stem(stem: str) -> str:
    if stem in SHORT_LABEL_MAP:
        return SHORT_LABEL_MAP[stem]
    if stem.startswith("auto_"):
        stem = stem[5:]
    tokens = stem.split("_")
    if len(tokens) <= 2:
        return stem[:16]
    abbr = "-".join(tok[:4] for tok in tokens[:3])
    return abbr[:16]


def _read_csv_summary(csv_path: Path, label: str, suite: str, is_baseline: bool) -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return None

    has_status = "status" in rows[0]
    ok_rows = rows if not has_status else [r for r in rows if (r.get("status", "") or "").strip().lower() == "ok"]

    summary = {
        "suite": suite,
        "label": label,
        "file": str(csv_path.relative_to(ROOT)),
        "num_rows": len(rows),
        "num_ok_rows": len(ok_rows),
        "is_baseline": is_baseline,
    }
    for m in METRICS:
        summary[m] = _mean(_to_float(r.get(m)) for r in ok_rows) if ok_rows else None
    return summary


def _collect_rows() -> list[dict]:
    rows: list[dict] = []

    for case in STATIC_CASES:
        row = _read_csv_summary(case.csv_path, case.label, case.suite, case.is_baseline)
        if row is not None:
            rows.append(row)

    all_feature_csvs = sorted(ALL_FEATURE_INPUT.glob("*.csv"))
    for csv_path in all_feature_csvs:
        stem = csv_path.stem
        label = _short_label_from_stem(stem)
        is_baseline = "baseline" in stem.lower()
        row = _read_csv_summary(csv_path, label, "all_features", is_baseline)
        if row is not None:
            rows.append(row)

    return rows


def _save_suite_bar(suite: str, metric: str, labels: list[str], values: list[float]) -> None:
    title = f"{SUITE_TITLES[suite]} - {METRIC_TITLE[metric]}"
    unit = METRIC_UNIT[metric]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(f"{unit} (lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / f"{suite}_{metric}.png", dpi=180)
    fig.savefig(FIGURE_ROOT / f"{suite}_{metric}.svg")
    plt.close(fig)


def _write_summary_files(rows: list[dict]) -> None:
    out_csv = PLOT_ROOT / "combined_summary.csv"
    fieldnames = [
        "suite",
        "label",
        "file",
        "num_rows",
        "num_ok_rows",
        "is_baseline",
        *METRICS,
        *(f"{m}_speedup_vs_baseline" for m in METRICS),
    ]
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            for m in METRICS:
                serial[m] = _fmt(serial.get(m))
                serial[f"{m}_speedup_vs_baseline"] = _fmt(serial.get(f"{m}_speedup_vs_baseline"))
            writer.writerow(serial)

    out_md = PLOT_ROOT / "combined_summary.md"
    lines: list[str] = [
        "# Experiments Summary",
        "",
        f"- Generated from {len(rows)} CSV files.",
        "- Each section uses its own baseline row for speedup.",
        "",
    ]

    for suite in ("grouped_compilation", "pipeline_only", "multi_gpu_benchmark", "all_features"):
        suite_rows = [r for r in rows if r["suite"] == suite]
        if not suite_rows:
            continue
        lines.extend(
            [
                f"## {SUITE_TITLES[suite]}",
                "",
                "| label | end_to_end_s | compilation_s | benchmark_s | best_latency_ms | e2e_speedup |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in suite_rows:
            lines.append(
                f"| {row['label']} | {_fmt(row.get('end_to_end_s'), 3)} | {_fmt(row.get('compilation_s'), 3)} | "
                f"{_fmt(row.get('benchmark_s'), 3)} | {_fmt(row.get('best_latency_ms'), 4)} | "
                f"{_fmt(row.get('end_to_end_s_speedup_vs_baseline'), 3)} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows()
    if not rows:
        print("No input CSV found. Rerun after experiments finish.")
        return 1

    # suite-wise baseline normalization
    for suite in {r["suite"] for r in rows}:
        suite_rows = [r for r in rows if r["suite"] == suite]
        baseline = next((r for r in suite_rows if r.get("is_baseline")), suite_rows[0])
        for row in suite_rows:
            for m in METRICS:
                row[f"{m}_speedup_vs_baseline"] = _safe_div(baseline.get(m), row.get(m))

    # draw per-suite bars (keeps old figure naming scheme)
    for suite in ("grouped_compilation", "pipeline_only", "multi_gpu_benchmark", "all_features"):
        suite_rows = [r for r in rows if r["suite"] == suite and r.get("num_ok_rows", 0) > 0]
        if not suite_rows:
            continue
        labels = [str(r["label"]) for r in suite_rows]
        for metric in METRICS:
            metric_rows = [r for r in suite_rows if r.get(metric) is not None]
            if not metric_rows:
                continue
            _save_suite_bar(
                suite=suite,
                metric=metric,
                labels=[str(r["label"]) for r in metric_rows],
                values=[float(r[metric]) for r in metric_rows],
            )

    _write_summary_files(rows)

    print(f"Wrote {PLOT_ROOT / 'combined_summary.csv'}")
    print(f"Wrote {PLOT_ROOT / 'combined_summary.md'}")
    print(f"Wrote figures to {FIGURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

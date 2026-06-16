#!/usr/bin/env python3
"""Summarize real experiment outputs for paper writing.

The script never edits raw results. It reads processed scheduler summaries (or
raw targeted summaries) and writes compact JSON/Markdown audits that highlight
the metrics needed by the paper: goodput, SLO attainment, tail TPOT, re-prefill
rate, KV evictions, selected k, and service-time breakdown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


KEY_METRICS = [
    "goodput_tok_s",
    "throughput_tok_s",
    "slo_attainment",
    "p99_tpot_ms",
    "reprefills_per_1k_tokens",
    "kv_evictions_per_1k_tokens",
    "mean_selected_k",
    "acceptance_rate",
    "service_target_frac",
    "service_draft_frac",
    "service_other_frac",
    "x_peak_gpu_mb",
]


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def find_scheduler_summaries(root: Path) -> List[Path]:
    processed = root / "processed" / "scheduler_all_summary.csv"
    if processed.exists():
        return [processed]
    raw = root / "raw"
    return [
        path
        for path in sorted(raw.glob("*/scheduler_*.csv"))
        if not path.name.startswith("scheduler_requests_")
        and not path.name.startswith("scheduler_calib_")
    ]


def pct(delta: float, base: float) -> float:
    return 100.0 * delta / base if base else float("nan")


def compare_main(df: pd.DataFrame) -> List[dict]:
    if df.empty or "policy" not in df or "N" not in df:
        return []
    if "kind" in df.columns:
        main = df[df["kind"].fillna("main").eq("main")].copy()
    else:
        main = df.copy()
    if "tag" in main:
        main = main[main["tag"].astype(str).str.contains("main|n64", case=False, regex=True)]
    rows = []
    for n_value, group in main.groupby("N"):
        by_policy = {str(row["policy"]): row for _, row in group.iterrows()}
        spec = by_policy.get("SpecSched")
        if spec is None:
            continue
        row = {"N": int(float(n_value)), "SpecSched_goodput": float(spec.get("goodput_tok_s", float("nan")))}
        for base in ["NoSpec", "FixedK-1", "FixedK-3", "TETRIS-Proxy"]:
            if base in by_policy:
                bg = float(by_policy[base].get("goodput_tok_s", 0.0))
                row[f"vs_{base}_pct"] = pct(float(spec.get("goodput_tok_s", 0.0)) - bg, bg)
        row["SpecSched_rep1000"] = float(spec.get("reprefills_per_1k_tokens", float("nan")))
        row["SpecSched_p99_tpot_ms"] = float(spec.get("p99_tpot_ms", float("nan")))
        rows.append(row)
    return rows


def markdown_summary(df: pd.DataFrame, comparisons: List[dict]) -> str:
    lines = ["# Experiment Result Audit", ""]
    if df.empty:
        lines.append("No scheduler summary rows found.")
        return "\n".join(lines) + "\n"

    cols = [c for c in ["tag", "kind", "policy", "N"] + KEY_METRICS if c in df.columns]
    shown = df[cols].copy()
    if "N" in shown:
        shown = shown.sort_values([c for c in ["tag", "N", "policy"] if c in shown.columns])
    lines.append(f"- Rows: {len(df)}")
    if "tag" in df:
        lines.append(f"- Tags: {', '.join(sorted(map(str, df['tag'].dropna().unique())))}")
    lines.append("")
    lines.append("## SpecSched Main Comparisons")
    lines.append("")
    if comparisons:
        lines.append(pd.DataFrame(comparisons).round(3).to_markdown(index=False))
    else:
        lines.append("No main SpecSched comparison rows found.")
    lines.append("")
    lines.append("## Key Rows")
    lines.append("")
    lines.append(shown.round(4).to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/local_v2")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csvs(find_scheduler_summaries(root))
    comparisons = compare_main(df)
    payload = {
        "results_root": str(root),
        "rows": int(len(df)),
        "comparisons": comparisons,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "audit.md").write_text(markdown_summary(df, comparisons), encoding="utf-8")
    print(out_dir / "audit.md")


if __name__ == "__main__":
    main()

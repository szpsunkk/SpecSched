#!/usr/bin/env python3
"""Analyze edge-draft bottlenecks from real scheduler CSV outputs.

The script uses only measured CSV files. It can run before the benchmark cell
finishes: calibration rows are enough to show whether edge drafting is compute
or communication dominated relative to the cloud-only NoSpec path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def read_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def find_csvs(root: Path, prefix: str) -> List[Path]:
    return sorted((root / "raw").glob(f"*/{prefix}*.csv"))


def numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def markdown(root: Path, calib: pd.DataFrame, summary: pd.DataFrame, requests: pd.DataFrame) -> str:
    lines = ["# Edge Bottleneck Audit", ""]
    lines.append(f"- Results root: `{root}`")
    lines.append("")

    if calib.empty:
        lines.append("No calibration CSV rows found yet.")
    else:
        calib = numeric(calib, ["k", "tokens_per_s", "acceptance", "tpot_ms", "ttft_ms"])
        by_k = (
            calib.groupby("k", dropna=False)
            .agg(
                rows=("k", "size"),
                mean_tok_s=("tokens_per_s", "mean"),
                median_tok_s=("tokens_per_s", "median"),
                mean_acceptance=("acceptance", "mean"),
                median_tpot_ms=("tpot_ms", "median"),
            )
            .reset_index()
            .sort_values("k")
        )
        lines.append("## Calibration Throughput By Draft Length")
        lines.append("")
        lines.append(by_k.round(3).to_markdown(index=False))
        lines.append("")

        base = by_k.loc[by_k["k"] == 0, "mean_tok_s"]
        spec = by_k.loc[by_k["k"] > 0, "mean_tok_s"]
        if not base.empty and not spec.empty:
            base_v = float(base.iloc[0])
            best_spec = float(spec.max())
            mean_spec = float(spec.mean())
            lines.append("## Edge Bottleneck Signal")
            lines.append("")
            lines.append(f"- Cloud-only NoSpec calibration throughput (k=0): `{base_v:.2f}` tok/s.")
            lines.append(f"- Best edge-draft calibration throughput (k>0): `{best_spec:.2f}` tok/s.")
            lines.append(f"- Mean edge-draft calibration throughput (k>0): `{mean_spec:.2f}` tok/s.")
            if base_v > 0:
                lines.append(f"- Best edge-draft path is `{base_v / max(best_spec, 1e-9):.2f}x` slower than k=0.")
                lines.append(f"- Average edge-draft path is `{base_v / max(mean_spec, 1e-9):.2f}x` slower than k=0.")
            lines.append("")

        if {"bucket", "k", "tokens_per_s"}.issubset(calib.columns):
            idx = calib.groupby("bucket")["tokens_per_s"].idxmax()
            best = calib.loc[idx, ["bucket", "k", "tokens_per_s", "acceptance"]].sort_values("bucket")
            lines.append("## Best Calibration k By Bucket")
            lines.append("")
            lines.append(best.round(3).to_markdown(index=False))
            lines.append("")

    if not summary.empty:
        keep = [
            "tag",
            "policy",
            "N",
            "goodput_tok_s",
            "throughput_tok_s",
            "slo_attainment",
            "p99_tpot_ms",
            "reprefills_per_1k_tokens",
            "kv_evictions_per_1k_tokens",
            "mean_selected_k",
            "service_target_frac",
            "service_draft_frac",
            "service_other_frac",
        ]
        cols = [col for col in keep if col in summary.columns]
        shown = numeric(summary[cols], [col for col in cols if col not in {"tag", "policy"}])
        lines.append("## Completed Scheduler Summary")
        lines.append("")
        lines.append(shown.round(4).to_markdown(index=False))
        lines.append("")

    if not requests.empty:
        req_cols = [col for col in ["policy", "N", "tpot_ms", "ttft_ms", "service_other_ms"] if col in requests.columns]
        if req_cols:
            req = numeric(requests[req_cols], [col for col in req_cols if col not in {"policy"}])
            agg = req.groupby([c for c in ["policy", "N"] if c in req.columns]).median(numeric_only=True).reset_index()
            lines.append("## Request-Level Medians")
            lines.append("")
            lines.append(agg.round(3).to_markdown(index=False))
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/edge_n64_specsched_fix")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = read_csvs(find_csvs(root, "scheduler_calib_"))
    summary = read_csvs([
        path for path in find_csvs(root, "scheduler_")
        if "scheduler_requests_" not in path.name and "scheduler_calib_" not in path.name
    ])
    requests = read_csvs(find_csvs(root, "scheduler_requests_"))

    payload = {
        "results_root": str(root),
        "calib_rows": int(len(calib)),
        "summary_rows": int(len(summary)),
        "request_rows": int(len(requests)),
    }
    (out_dir / "edge_bottleneck.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "edge_bottleneck.md").write_text(markdown(root, calib, summary, requests), encoding="utf-8")
    print(out_dir / "edge_bottleneck.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize measured non-stationary transfer GPU runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from .config_utils import load_config as load_config_file
except ImportError:
    from config_utils import load_config as load_config_file


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_LABELS = {
    "shift": "Workload shift",
    "bursty": "Bursty arrivals",
    "tiered": "Tiered SLOs",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SUMMARY_FIELDS = [
    "scenario",
    "tag",
    "N",
    "bestfixed_policy",
    "bestfixed_goodput_tok_s",
    "specsched_goodput_tok_s",
    "specsched_lift_vs_bestfixed_pct",
    "bestfixed_p99_tpot_ms",
    "specsched_p99_tpot_ms",
    "specsched_slo_attainment",
    "source",
    "noslo_goodput_tok_s",
    "specsched_lift_vs_noslo_pct",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def scenario_from_tag(tag: str) -> str:
    for key, label in SCENARIO_LABELS.items():
        if key in tag:
            return label
    return tag


def summarize(rows: list[dict]) -> list[dict]:
    by_tag_n: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_tag_n.setdefault((str(row.get("tag", "")), str(row.get("N", ""))), []).append(row)

    out: list[dict] = []
    for (tag, n_value), group in sorted(by_tag_n.items()):
        fixed = [row for row in group if str(row.get("policy", "")).startswith("FixedK-")]
        spec = next((row for row in group if row.get("policy") == "SpecSched"), None)
        noslo = next((row for row in group if row.get("policy") == "SpecSched-NoSLO"), None)
        if not fixed or spec is None:
            continue
        best = max(fixed, key=lambda row: as_float(row, "goodput_tok_s", -math.inf))
        best_goodput = as_float(best, "goodput_tok_s", 0.0)
        spec_goodput = as_float(spec, "goodput_tok_s", 0.0)
        row = {
            "scenario": scenario_from_tag(tag),
            "tag": tag,
            "N": n_value,
            "bestfixed_policy": best.get("policy", ""),
            "bestfixed_goodput_tok_s": f"{best_goodput:.4f}",
            "specsched_goodput_tok_s": f"{spec_goodput:.4f}",
            "specsched_lift_vs_bestfixed_pct": f"{((spec_goodput / best_goodput - 1.0) * 100.0) if best_goodput > 0 else float('nan'):.4f}",
            "bestfixed_p99_tpot_ms": f"{as_float(best, 'p99_tpot_ms'):.4f}",
            "specsched_p99_tpot_ms": f"{as_float(spec, 'p99_tpot_ms'):.4f}",
            "specsched_slo_attainment": f"{as_float(spec, 'slo_attainment'):.4f}",
            "source": "measured_gpu",
        }
        if noslo is not None:
            noslo_goodput = as_float(noslo, "goodput_tok_s", 0.0)
            row["noslo_goodput_tok_s"] = f"{noslo_goodput:.4f}"
            row["specsched_lift_vs_noslo_pct"] = f"{((spec_goodput / noslo_goodput - 1.0) * 100.0) if noslo_goodput > 0 else float('nan'):.4f}"
        out.append(row)
    return out


def write_latex(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Measured non-stationary transfer replay summary. Values are emitted only after GPU runs complete.}",
        "\\label{tab:transfer-gpu-measured}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scenario & BestFixed & SpecSched & Lift & P99 TPOT \\\\",
        "\\midrule",
    ]
    if not rows:
        lines.append("\\multicolumn{5}{c}{No completed GPU transfer rows yet.} \\\\")
    for row in rows:
        lines.append(
            f"{row['scenario']} & {float(row['bestfixed_goodput_tok_s']):.1f} & "
            f"{float(row['specsched_goodput_tok_s']):.1f} & "
            f"{float(row['specsched_lift_vs_bestfixed_pct']):+.1f}\\% & "
            f"{float(row['specsched_p99_tpot_ms']):.0f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper3090_transfer_gpu.json")
    args = parser.parse_args()

    cfg = load_config_file(Path(args.config).resolve(), ROOT).cfg
    root = Path(cfg["paths"]["results_root"]).resolve()
    processed = root / "processed"
    rows = read_csv(processed / "scheduler_transfer_gpu_summary.csv")
    summary = summarize(rows)
    write_csv(processed / "transfer_gpu_bestfixed_summary.csv", summary, SUMMARY_FIELDS)
    write_latex(ROOT / "paper_assets" / "tables" / "transfer_gpu_measured.tex", summary)
    manifest = {
        "source": str(processed / "scheduler_transfer_gpu_summary.csv"),
        "rows": len(summary),
        "scope": "Measured GPU transfer replay summary. Empty or smoke rows must not be used as paper headline evidence.",
    }
    (processed / "transfer_gpu_processing_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote transfer GPU summary to {processed}")


if __name__ == "__main__":
    main()

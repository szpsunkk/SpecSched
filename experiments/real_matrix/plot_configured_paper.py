"""Generate paper figures from processed configured experiment CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from .config_utils import load_config as load_config_file
except ImportError:
    from config_utils import load_config as load_config_file


ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "NoSpec": "#4d4d4d",
    "FixedK-1": "#bdbdbd",
    "FixedK-3": "#b2182b",
    "FixedK-5": "#ef8a62",
    "FixedK-7": "#fddbc7",
    "IndepOpt": "#2166ac",
    "TETRIS-Proxy": "#762a83",
    "SpecSched": "#1a9850",
    "SpecSched-NoBatch": "#a6d96a",
    "SpecSched-NoEvict": "#fdae61",
    "SpecSched-NoSLO": "#d73027",
    "SpecSched-LRU": "#fdae61",
    "SpecSched-Longest": "#9970ab",
    "SpecSched-H2O": "#80cdc1",
    "SpecSched-vLLM": "#e08214"
}
MARKERS = {
    "NoSpec": "o",
    "FixedK-1": "s",
    "FixedK-3": "^",
    "FixedK-5": "v",
    "FixedK-7": "D",
    "IndepOpt": "P",
    "TETRIS-Proxy": "X",
    "SpecSched": "*",
    "SpecSched-NoBatch": "<",
    "SpecSched-NoEvict": ">",
    "SpecSched-NoSLO": "h",
    "SpecSched-LRU": ">",
    "SpecSched-Longest": "8",
    "SpecSched-H2O": "p",
    "SpecSched-vLLM": "d"
}


def load_config(path: Path) -> dict:
    return load_config_file(path, ROOT).cfg


def numeric(value):
    if value is None or value == "":
        return value
    try:
        text = str(value)
        if "." in text or "e" in text.lower() or text.lower() in ("nan", "inf"):
            return float(text)
        return int(text)
    except ValueError:
        return value


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: numeric(v) for k, v in row.items()} for row in csv.DictReader(f)]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })


def is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def ordered_policies(rows: List[dict]) -> List[str]:
    policies = []
    for row in rows:
        policy = str(row.get("policy", ""))
        if policy and policy not in policies:
            policies.append(policy)
    return policies


def save(fig, outdir: Path, name: str) -> None:
    fig.tight_layout(pad=0.45)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def metric_by_n(rows: List[dict], policy: str, metric: str, ns: List[int]) -> List[float]:
    values: Dict[int, float] = {}
    for row in rows:
        if row.get("policy") == policy and is_finite(row.get(metric)):
            values[int(row["N"])] = float(row[metric])
    return [values.get(n, float("nan")) for n in ns]


def line_vs_n(rows: List[dict], outdir: Path, metric: str, ylabel: str, name: str, logy: bool = False) -> None:
    if not rows:
        return
    ns = sorted({int(r["N"]) for r in rows if r.get("N") != ""})
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.35))
    for policy in ordered_policies(rows):
        vals = metric_by_n(rows, policy, metric, ns)
        ax.plot(ns, vals, marker=MARKERS.get(policy, "o"), linewidth=1.2, markersize=3.5,
                color=COLORS.get(policy, "#333333"), label=policy)
    if len(ns) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Concurrent users N")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.75)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, loc="best", ncol=2 if len(ordered_policies(rows)) > 4 else 1)
    save(fig, outdir, name)


def metric_by_budget(rows: List[dict], policy: str, metric: str, budgets: List[int], n_value: int) -> List[float]:
    values: Dict[int, float] = {}
    for row in rows:
        if row.get("policy") == policy and int(row.get("N", -1)) == n_value and is_finite(row.get(metric)):
            values[int(row["configured_kv_blocks"])] = float(row[metric])
    return [values.get(b, float("nan")) for b in budgets]


def line_vs_budget(rows: List[dict], outdir: Path, metric: str, ylabel: str, name: str, logy: bool = False) -> None:
    if not rows:
        return
    n_value = max(int(r["N"]) for r in rows if r.get("N") != "")
    budgets = sorted({int(r["configured_kv_blocks"]) for r in rows if r.get("configured_kv_blocks") != ""})
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.35))
    for policy in ordered_policies(rows):
        vals = metric_by_budget(rows, policy, metric, budgets, n_value)
        ax.plot(budgets, vals, marker=MARKERS.get(policy, "o"), linewidth=1.2, markersize=3.5,
                color=COLORS.get(policy, "#333333"), label=policy)
    ax.set_xlabel("KV block budget")
    ax.set_ylabel(ylabel)
    ax.text(0.02, 0.98, f"N={n_value}", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.75)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, loc="best")
    save(fig, outdir, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper3090.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    root = Path(cfg["paths"]["results_root"]).resolve()
    processed = root / "processed"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    setup()

    main_rows = read_csv(processed / "scheduler_main_summary.csv")
    kv_rows = read_csv(processed / "scheduler_kv_sweep_summary.csv")
    line_vs_n(main_rows, figures, "goodput_tok_s", "SLO goodput (tok/s)", "real_scheduler_goodput_lines")
    line_vs_n(main_rows, figures, "slo_attainment", "SLO attainment", "real_scheduler_attainment_lines")
    line_vs_n(main_rows, figures, "p99_tpot_ms", "P99 TPOT (ms/token)", "real_scheduler_p99_tpot_lines", logy=True)
    line_vs_n(main_rows, figures, "kv_evictions", "KV evictions", "real_scheduler_evictions_lines")
    line_vs_budget(kv_rows, figures, "goodput_tok_s", "SLO goodput (tok/s)", "kv_budget_goodput")
    line_vs_budget(kv_rows, figures, "p99_tpot_ms", "P99 TPOT (ms/token)", "kv_budget_p99_tpot", logy=True)
    line_vs_budget(kv_rows, figures, "reprefills_per_1k_tokens", "Re-prefills / 1K tokens", "kv_budget_refill")
    line_vs_budget(kv_rows, figures, "kv_evictions_per_1k_tokens", "KV evictions / 1K tokens", "kv_budget_evictions")
    print(f"saved figures to {figures}")


if __name__ == "__main__":
    main()

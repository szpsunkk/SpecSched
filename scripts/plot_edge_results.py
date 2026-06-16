#!/usr/bin/env python3
"""Generate compact paper assets for the real Jetson edge experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

POLICIES = ["NoSpec", "FixedK-1", "FixedK-3", "IndepOpt", "TETRIS-Proxy", "SpecSched", "SpecSched-Safe"]
COLORS = {
    "NoSpec": "#2B2B2B",
    "FixedK-1": "#1F4E79",
    "FixedK-3": "#2F6F9F",
    "IndepOpt": "#5E3C99",
    "TETRIS-Proxy": "#9C4F00",
    "SpecSched": "#006D5B",
    "SpecSched-Safe": "#8C2D04",
}
HATCHES = {
    "NoSpec": "",
    "FixedK-1": "//",
    "FixedK-3": "\\\\",
    "IndepOpt": "++",
    "TETRIS-Proxy": "**",
    "SpecSched": "",
    "SpecSched-Safe": "xx",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.85,
            "axes.axisbelow": True,
            "grid.linewidth": 0.35,
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def ordered(df: pd.DataFrame) -> list[str]:
    present = set(df["policy"])
    return [p for p in POLICIES if p in present]


def metric_value(row: pd.Series, metric: str) -> float:
    mean_key = f"{metric}_mean"
    if mean_key in row:
        return float(row[mean_key])
    return float(row[metric])


def metric_std(row: pd.Series, metric: str) -> float:
    std_key = f"{metric}_std"
    if std_key in row:
        return float(row[std_key])
    return 0.0


def load_edge_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    multiseed = root / "processed" / "main_multiseed_summary.csv"
    if multiseed.exists():
        plot_df = pd.read_csv(multiseed)
        plot_df["tag"] = "edge_main_beta15"
        raw_breakdown = root / "seed_7" / "processed" / "scheduler_main_summary.csv"
        breakdown_df = pd.read_csv(raw_breakdown) if raw_breakdown.exists() else plot_df.copy()
        return plot_df, breakdown_df
    df = pd.read_csv(root / "processed" / "scheduler_all_summary.csv")
    return df, df.copy()


def edge_main_plot(df: pd.DataFrame, outdir: Path) -> None:
    main = df[df["tag"].eq("edge_main_beta15") & df["policy"].isin(POLICIES)].copy()
    policies = ordered(main)
    ns = sorted(main["N"].unique())
    x_base = range(len(ns))
    width = 0.13
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9))
    third_metric = "throughput_tok_s" if (
        "throughput_tok_s" in main.columns or "throughput_tok_s_mean" in main.columns
    ) else "slo_attainment"
    third_label = "Throughput\n(tokens/s)" if third_metric == "throughput_tok_s" else "SLO attainment"
    metrics = [
        ("goodput_tok_s", "SLO goodput\n(tokens/s)", False),
        ("p99_tpot_ms", "P99 TPOT\n(ms/token)", True),
        (third_metric, third_label, False),
    ]
    offsets = [(i - (len(policies) - 1) / 2) * width for i in range(len(policies))]
    for ax, (metric, ylabel, log_y) in zip(axes, metrics):
        for offset, policy in zip(offsets, policies):
            vals = []
            yerr = []
            for n in ns:
                row = main[(main["N"].eq(n)) & (main["policy"].eq(policy))]
                if row.empty:
                    vals.append(0.0)
                    yerr.append(0.0)
                else:
                    vals.append(metric_value(row.iloc[0], metric))
                    yerr.append(metric_std(row.iloc[0], metric))
            bars = ax.bar(
                [x + offset for x in x_base],
                vals,
                yerr=yerr if any(v > 0 for v in yerr) else None,
                width=width,
                label=policy,
                color=COLORS.get(policy, "#777777"),
                edgecolor="#111111",
                linewidth=0.45,
                hatch=HATCHES.get(policy, ""),
                alpha=0.95,
                error_kw={"elinewidth": 0.55, "capsize": 1.5, "capthick": 0.55},
            )
            if metric == "goodput_tok_s":
                for bar, val in zip(bars, vals):
                    if val == 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            0.9,
                            "0",
                            ha="center",
                            va="bottom",
                            fontsize=6.2,
                            color="#333333",
                        )
        ax.set_xticks(list(x_base))
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle=":", color="#B8B8B8")
        for spine in ax.spines.values():
            spine.set_visible(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=6,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.4,
    )
    fig.tight_layout(w_pad=0.7)
    save(fig, outdir, "edge_main_bottleneck")


def edge_breakdown_plot(df: pd.DataFrame, outdir: Path) -> None:
    main = df[df["tag"].eq("edge_main_beta15") & df["N"].eq(8) & df["policy"].isin(POLICIES)].copy()
    policies = ordered(main)
    main = main.set_index("policy").loc[policies].reset_index()
    comps = [
        ("service_target_frac", "Target", "#2B2B2B"),
        ("service_draft_frac", "Draft", "#006D5B"),
        ("service_other_frac", "Network/other", "#9C4F00"),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    bottom = [0.0] * len(main)
    for col, label, color in comps:
        vals = [float(v) for v in main[col]]
        ax.bar(
            range(len(main)),
            vals,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="#111111",
            linewidth=0.45,
            alpha=0.92,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(main)))
    ax.set_xticklabels(policies, rotation=28, ha="right")
    ax.set_ylabel("Service-time fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", linestyle=":", color="#B8B8B8")
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, outdir, "edge_service_breakdown_n8")


def write_tables_and_summary(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    main = df[df["tag"].eq("edge_main_beta15") & df["policy"].isin(POLICIES)].copy()
    cols = [
        "policy",
        "N",
        "goodput_tok_s_mean",
        "goodput_tok_s_std",
        "p99_tpot_ms_mean",
        "p99_tpot_ms_std",
        "slo_attainment_mean",
        "slo_attainment_std",
    ]
    if "goodput_tok_s_mean" not in main.columns:
        rename = {
            "goodput_tok_s": "goodput_tok_s_mean",
            "p99_tpot_ms": "p99_tpot_ms_mean",
            "slo_attainment": "slo_attainment_mean",
        }
        table = main.rename(columns=rename).copy()
        for col in ("goodput_tok_s_std", "p99_tpot_ms_std", "slo_attainment_std"):
            table[col] = 0.0
        table = table[cols]
    else:
        table = main[cols].copy()
    table = table.sort_values(["N", "policy"])
    table.to_csv(outdir / "edge_main_summary.csv", index=False)
    tex_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Policy & $N$ & Goodput & Attain. & P99 TPOT \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        tex_lines.append(
            f"{row.policy} & {int(row.N)} & {row.goodput_tok_s_mean:.2f} & "
            f"{row.slo_attainment_mean:.2f} & {row.p99_tpot_ms_mean:.1f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (outdir / "edge_main_summary.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    by = {(str(r.policy), int(r.N)): r for r in table.itertuples(index=False)}
    nospec4 = by[("NoSpec", 4)]
    nospec8 = by[("NoSpec", 8)]
    spec4 = by[("SpecSched", 4)]
    spec8 = by[("SpecSched", 8)]
    safe4 = by.get(("SpecSched-Safe", 4))
    safe8 = by.get(("SpecSched-Safe", 8))
    payload = {
        "nospec_goodput_n4": float(nospec4.goodput_tok_s_mean),
        "nospec_goodput_n8": float(nospec8.goodput_tok_s_mean),
        "specsched_goodput_n4": float(spec4.goodput_tok_s_mean),
        "specsched_goodput_n8": float(spec8.goodput_tok_s_mean),
        "safe_goodput_n4": float(safe4.goodput_tok_s_mean) if safe4 else None,
        "safe_goodput_n8": float(safe8.goodput_tok_s_mean) if safe8 else None,
        "specsched_p99_n8": float(spec8.p99_tpot_ms_mean),
        "nospec_p99_n8": float(nospec8.p99_tpot_ms_mean),
        "safe_p99_n8": float(safe8.p99_tpot_ms_mean) if safe8 else None,
    }
    (outdir / "edge_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# Edge Practical Result Summary",
        "",
        "- Real setup: RTX 3090 Ti target plus three Jetson Orin draft workers over Tailscale/SSH.",
        f"- NoSpec reaches {payload['nospec_goodput_n4']:.2f} tok/s at N=4 and {payload['nospec_goodput_n8']:.2f} tok/s at N=8.",
        f"- Unprotected SpecSched has 0 SLO goodput at both N=4 and N=8; its N=8 P99 TPOT is {payload['specsched_p99_n8']:.1f} ms/token versus NoSpec {payload['nospec_p99_n8']:.1f} ms/token.",
        f"- SpecSched-Safe falls back to NoSpec and reaches {payload['safe_goodput_n4']:.2f} tok/s at N=4 and {payload['safe_goodput_n8']:.2f} tok/s at N=8.",
        "- The service-time breakdown shows edge draft plus network/SSH overhead dominates unprotected speculative policies.",
        "",
    ]
    (outdir / "edge_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/edge_practical")
    parser.add_argument("--figures-dir", default="paper_assets/figures")
    parser.add_argument("--tables-dir", default="paper_assets/tables")
    args = parser.parse_args()
    root = Path(args.results_root)
    if not root.is_absolute():
        root = ROOT / root
    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = ROOT / figures_dir
    tables_dir = Path(args.tables_dir)
    if not tables_dir.is_absolute():
        tables_dir = ROOT / tables_dir
    configure_matplotlib()
    df, breakdown_df = load_edge_frames(root)
    for col in [
        "N",
        "goodput_tok_s",
        "goodput_tok_s_mean",
        "goodput_tok_s_std",
        "throughput_tok_s",
        "throughput_tok_s_mean",
        "throughput_tok_s_std",
        "slo_attainment",
        "slo_attainment_mean",
        "slo_attainment_std",
        "p99_tpot_ms",
        "p99_tpot_ms_mean",
        "p99_tpot_ms_std",
        "acceptance_rate",
        "service_target_frac",
        "service_draft_frac",
        "service_other_frac",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in breakdown_df:
            breakdown_df[col] = pd.to_numeric(breakdown_df[col], errors="coerce")
    edge_main_plot(df, figures_dir)
    edge_breakdown_plot(breakdown_df, figures_dir)
    write_tables_and_summary(df, tables_dir)
    print(json.dumps({
        "figures": [
            str(figures_dir / "edge_main_bottleneck.pdf"),
            str(figures_dir / "edge_service_breakdown_n8.pdf"),
        ],
        "tables": [
            str(tables_dir / "edge_main_summary.csv"),
            str(tables_dir / "edge_summary.md"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

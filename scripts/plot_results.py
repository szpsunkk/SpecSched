#!/usr/bin/env python3
"""Generate IEEE/ACM-style paper figures and tables from processed results.

The script does not alter measured values. It only filters, groups, formats, and
writes plots/tables derived from processed CSV files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

MAIN_POLICIES = [
    "NoSpec",
    "FixedK-1",
    "FixedK-7",
    "SpecSched",
    "TETRIS-Proxy",
]

KV_POLICIES = [
    "SpecSched",
    "SpecSched-vLLM",
    "SpecSched-LRU",
    "SpecSched-Longest",
    "SpecSched-H2O",
]

SLO_POLICIES = [
    "IndepOpt",
    "SpecSched",
]

ABLATION_POLICIES = [
    "SpecSched",
    "SpecSched-NoBatch",
    "SpecSched-NoEvict",
    "SpecSched-NoSLO",
]

PAPER_POLICIES = list(dict.fromkeys(MAIN_POLICIES + KV_POLICIES + SLO_POLICIES + ABLATION_POLICIES))

POLICY_ORDER = [
    "NoSpec",
    "FixedK-1",
    "FixedK-3",
    "FixedK-5",
    "FixedK-7",
    "IndepOpt",
    "TETRIS-Proxy",
    "SpecSched",
    "SpecSched-vLLM",
    "SpecSched-LRU",
    "SpecSched-Longest",
    "SpecSched-H2O",
    "SpecSched-NoBatch",
    "SpecSched-NoEvict",
    "SpecSched-NoSLO",
]

COLORS = {
    "NoSpec": "#2B2B2B",
    "Best FixedK": "#1F4E79",
    "Best alternative": "#4B5563",
    "LRU": "#B45F06",
    "FixedK-1": "#1F4E79",
    "FixedK-3": "#2F6F9F",
    "FixedK-5": "#4A7FA7",
    "FixedK-7": "#566C8D",
    "IndepOpt": "#5E3C99",
    "TETRIS-Proxy": "#9C4F00",
    "SpecSched": "#006D5B",
    "SpecSched-vLLM": "#0072B2",
    "SpecSched-LRU": "#B45F06",
    "SpecSched-Longest": "#7B3294",
    "SpecSched-H2O": "#4D7C0F",
    "SpecSched-NoBatch": "#0B4F6C",
    "SpecSched-NoEvict": "#8C6D1F",
    "SpecSched-NoSLO": "#9D1F2D",
}

MARKERS = {
    "NoSpec": "o",
    "Best FixedK": "D",
    "Best alternative": "o",
    "LRU": ">",
    "FixedK-1": "s",
    "FixedK-3": "^",
    "FixedK-5": "v",
    "FixedK-7": "D",
    "IndepOpt": "P",
    "TETRIS-Proxy": "X",
    "SpecSched": "*",
    "SpecSched-vLLM": "d",
    "SpecSched-LRU": ">",
    "SpecSched-Longest": "8",
    "SpecSched-H2O": "p",
    "SpecSched-NoBatch": "<",
    "SpecSched-NoEvict": "h",
    "SpecSched-NoSLO": "H",
}

LINESTYLES = {
    "NoSpec": (0, (3.0, 1.4)),
    "Best FixedK": "-",
    "Best alternative": (0, (4.5, 1.4)),
    "LRU": (0, (1.4, 1.2)),
    "FixedK-1": (0, (1.2, 1.2)),
    "FixedK-3": (0, (1.2, 1.2)),
    "FixedK-5": (0, (1.2, 1.2)),
    "FixedK-7": (0, (5.0, 1.5)),
    "IndepOpt": (0, (4.0, 1.4, 1.0, 1.4)),
    "TETRIS-Proxy": (0, (6.0, 1.6)),
    "SpecSched": "-",
    "SpecSched-vLLM": (0, (3.0, 1.2)),
    "SpecSched-LRU": (0, (1.2, 1.2)),
    "SpecSched-Longest": (0, (5.0, 1.5)),
    "SpecSched-H2O": (0, (4.0, 1.4, 1.0, 1.4)),
    "SpecSched-NoBatch": (0, (3.0, 1.2)),
    "SpecSched-NoEvict": (0, (5.0, 1.5)),
    "SpecSched-NoSLO": (0, (1.2, 1.2)),
}

HATCHES = {
    "NoSpec": "",
    "FixedK-1": "//",
    "FixedK-3": "\\\\",
    "FixedK-5": "xx",
    "FixedK-7": "..",
    "IndepOpt": "++",
    "TETRIS-Proxy": "**",
    "SpecSched": "",
    "SpecSched-vLLM": "//",
    "SpecSched-LRU": "\\\\",
    "SpecSched-Longest": "xx",
    "SpecSched-H2O": "..",
    "SpecSched-NoBatch": "//",
    "SpecSched-NoEvict": "\\\\",
    "SpecSched-NoSLO": "xx",
}

TEAL_CMAP = LinearSegmentedColormap.from_list(
    "specsched_teal",
    ["#F7F9F8", "#C8DAD6", "#6FA79B", "#006D5B"],
)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "specsched_blue",
    ["#F8F9FB", "#C7D1E0", "#6F86A8", "#1F4E79"],
)
AMBER_CMAP = LinearSegmentedColormap.from_list(
    "specsched_amber",
    ["#F9F7F2", "#E4C99A", "#C0873F", "#8A4B08"],
)

SUMMARY_FILENAMES = ["summary.csv", "scheduler_all_summary.csv"]
OVERHEAD_SUMMARY = ROOT / "results/scheduler_overhead_micro/processed/scheduler_all_summary.csv"

SPLIT_FILENAMES = {
    "main": "scheduler_main_summary.csv",
    "kv_sweep": "scheduler_kv_sweep_summary.csv",
    "slo_sensitivity": "scheduler_slo_sensitivity_summary.csv",
    "ablation": "ablation_summary.csv",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.85,
            "axes.axisbelow": True,
            "lines.linewidth": 1.55,
            "lines.markersize": 4.6,
            "grid.linewidth": 0.38,
            "hatch.linewidth": 0.55,
            "legend.handlelength": 1.85,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def processed_dir_from_results_root(results_root: str | None) -> Path:
    if not results_root:
        return resolve("results/processed")
    root = resolve(results_root)
    return root / "processed"


def load_summary(summary_arg: str | None, processed_dir: Path) -> tuple[pd.DataFrame, Path]:
    candidates = [summary_arg] if summary_arg else [processed_dir / name for name in SUMMARY_FILENAMES]
    checked: List[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = resolve(candidate)
        checked.append(str(path))
        if path.exists():
            df = pd.read_csv(path)
            return normalize(df), path
    raise FileNotFoundError("No summary CSV found. Checked: " + ", ".join(checked))


def load_split(kind: str, all_rows: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / SPLIT_FILENAMES[kind]
    if path.exists():
        return normalize(pd.read_csv(path))
    if "kind" not in all_rows.columns:
        return pd.DataFrame()
    return all_rows[all_rows["kind"] == kind].copy()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "N",
        "requests",
        "tokens",
        "wall_s",
        "throughput_tok_s",
        "goodput_tok_s",
        "slo_attainment",
        "p99_tpot_ms",
        "mean_tpot_ms",
        "mean_ttft_ms",
        "acceptance_rate",
        "target_forwards_per_token",
        "tokens_per_target_forward",
        "reprefills_per_1k_tokens",
        "kv_evictions",
        "kv_evictions_per_1k_tokens",
        "configured_beta",
        "configured_kv_blocks",
        "x_peak_gpu_mb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ordered_policies(df: pd.DataFrame, policies: Sequence[str] | None = None) -> List[str]:
    present = set(df["policy"].dropna())
    order = list(policies) if policies else POLICY_ORDER
    return [p for p in order if p in present]


def filter_paper_policies(df: pd.DataFrame, policies: Sequence[str] | None = None) -> pd.DataFrame:
    if df.empty or "policy" not in df.columns:
        return df.copy()
    keep = set(policies or PAPER_POLICIES)
    return df[df["policy"].isin(keep)].copy()


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def save_prelaid_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=(0, (1.0, 2.0)), color="#B8B8B8", alpha=0.82)
    ax.grid(True, axis="x", linestyle=(0, (1.0, 2.0)), color="#D0D0D0", alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.85)
    ax.tick_params(direction="out", length=3.2, width=0.75, colors="#111111", pad=2.0)


def style_legend(ax: plt.Axes, *, ncol: int, **kwargs) -> None:
    legend_kwargs = {
        "borderpad": 0.28,
        "labelspacing": 0.28,
    }
    legend_kwargs.update(kwargs)
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#BDBDBD",
        **legend_kwargs,
    )
    if legend:
        legend.get_frame().set_linewidth(0.45)


def plot_policy_line(ax: plt.Axes, x_values, y_values, policy: str, *, highlight: str = "SpecSched") -> None:
    is_focus = policy == highlight
    ax.plot(
        x_values,
        y_values,
        label=policy,
        marker=MARKERS.get(policy, "o"),
        color=COLORS.get(policy, "#333333"),
        linestyle=LINESTYLES.get(policy, "-"),
        linewidth=2.0 if is_focus else 1.18,
        markersize=5.2 if is_focus else 3.9,
        alpha=1.0 if is_focus else 0.72,
        markerfacecolor=COLORS.get(policy, "#333333"),
        markeredgecolor="white",
        markeredgewidth=0.48,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=5 if is_focus else 3,
    )


def plot_series_line(
    ax: plt.Axes,
    x_values,
    y_values,
    label: str,
    *,
    color: str | None = None,
    marker: str | None = None,
    linestyle=None,
    focus: bool = False,
    alpha: float | None = None,
) -> None:
    color = color or COLORS.get(label, "#333333")
    marker = marker or MARKERS.get(label, "o")
    linestyle = linestyle if linestyle is not None else LINESTYLES.get(label, "-")
    ax.plot(
        x_values,
        y_values,
        label=label,
        marker=marker,
        color=color,
        linestyle=linestyle,
        linewidth=2.25 if focus else 1.35,
        markersize=5.2 if focus else 4.0,
        alpha=1.0 if alpha is None and focus else (0.78 if alpha is None else alpha),
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=6 if focus else 3,
    )


def direct_label_last(
    ax: plt.Axes,
    x_values,
    y_values,
    label: str,
    *,
    color: str,
    dy: float = 0.0,
    fmt: str | None = None,
) -> None:
    if len(x_values) == 0:
        return
    x = list(x_values)[-1]
    y = list(y_values)[-1]
    text = fmt.format(y) if fmt else label
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.8,
        color=color,
        clip_on=False,
    )


def line_vs_n(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    outdir: Path,
    name: str,
    policies: Sequence[str] | None = None,
    log_y: bool = False,
) -> None:
    df = df.dropna(subset=["N", "policy", metric])
    if df.empty:
        return
    ns = sorted(int(n) for n in df["N"].dropna().unique())
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for policy in ordered_policies(df, policies):
        sub = df[df["policy"] == policy].sort_values("N")
        if sub.empty:
            continue
        plot_policy_line(ax, sub["N"].tolist(), sub[metric].tolist(), policy)
    if len(ns) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Concurrent requests (N)")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    style_axis(ax)
    style_legend(ax, ncol=3)
    save_figure(fig, outdir, name)


def line_grid_vs_n(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
    policies: Sequence[str] | None = None,
) -> None:
    metrics = [
        ("goodput_tok_s", "SLO goodput (tok/s)", "(a) SLO-aware goodput", False),
        ("slo_attainment", "SLO attainment", "(b) SLO attainment", False),
        ("p99_tpot_ms", "P99 TPOT (ms/token)", "(c) Tail TPOT", True),
        ("throughput_tok_s", "Throughput (tok/s)", "(d) Raw throughput", False),
    ]
    needed = ["N", "policy"] + [m[0] for m in metrics]
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    if df.empty:
        return
    ns = sorted(int(n) for n in df["N"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.7), sharex=True)
    handles = labels = None
    for ax, (metric, ylabel, title, log_y) in zip(axes.ravel(), metrics):
        for policy in ordered_policies(df, policies):
            sub = df[df["policy"] == policy].sort_values("N")
            if sub.empty:
                continue
            plot_policy_line(ax, sub["N"].tolist(), sub[metric].tolist(), policy)
        if len(ns) > 1:
            ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_title(title, loc="left", pad=2.0, fontweight="bold")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        style_axis(ax)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for ax in axes[-1, :]:
        ax.set_xlabel("Concurrent requests (N)")
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(len(labels), 5),
            frameon=False,
            columnspacing=0.95,
            handlelength=1.65,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.45, w_pad=0.72, h_pad=0.58)
    save_prelaid_figure(fig, outdir, name)


def main_clean_lines(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
) -> None:
    df = df.dropna(subset=["N", "policy", "goodput_tok_s", "p99_tpot_ms"])
    if df.empty:
        return
    ns = sorted(int(n) for n in df["N"].dropna().unique())
    fixed = df[df["policy"].isin(["FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7"])]
    best_good = fixed.loc[fixed.groupby("N")["goodput_tok_s"].idxmax()][["N", "goodput_tok_s"]].sort_values("N")
    best_p99 = fixed.loc[fixed.groupby("N")["p99_tpot_ms"].idxmin()][["N", "p99_tpot_ms"]].sort_values("N")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.42), sharex=True)
    series = [
        ("NoSpec", "NoSpec", "#2B2B2B", (0, (3.0, 1.4)), "o"),
        ("Best FixedK", None, "#1F4E79", "-", "D"),
        ("TETRIS-Proxy", "TETRIS-Proxy", "#9C4F00", (0, (5.5, 1.5)), "X"),
        ("SpecSched", "SpecSched", "#006D5B", "-", "*"),
    ]

    ax = axes[0]
    for label, policy, color, linestyle, marker in series:
        if label == "Best FixedK":
            x, y = best_good["N"].tolist(), best_good["goodput_tok_s"].tolist()
        else:
            sub = df[df["policy"] == policy].sort_values("N")
            x, y = sub["N"].tolist(), sub["goodput_tok_s"].tolist()
        plot_series_line(ax, x, y, label, color=color, marker=marker, linestyle=linestyle, focus=(label == "SpecSched"))
    ax.set_title("(a) SLO-aware goodput", loc="left", pad=2.0, fontweight="bold")
    ax.set_ylabel("Goodput (tok/s)")

    ax = axes[1]
    for label, policy, color, linestyle, marker in series:
        if label == "Best FixedK":
            x, y = best_p99["N"].tolist(), best_p99["p99_tpot_ms"].tolist()
        else:
            sub = df[df["policy"] == policy].sort_values("N")
            x, y = sub["N"].tolist(), sub["p99_tpot_ms"].tolist()
        plot_series_line(ax, x, y, label, color=color, marker=marker, linestyle=linestyle, focus=(label == "SpecSched"))
    ax.set_title("(b) Tail TPOT", loc="left", pad=2.0, fontweight="bold")
    ax.set_ylabel("P99 TPOT (ms/token)")
    ax.set_yscale("log")

    for ax in axes:
        if len(ns) > 1:
            ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Concurrent requests (N)")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.7,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.35, w_pad=0.85)
    save_prelaid_figure(fig, outdir, name)


def matrix_from_pivot(
    df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    row_order: Sequence[str] | None = None,
    col_order: Sequence[int] | None = None,
) -> pd.DataFrame:
    matrix = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    if row_order is not None:
        matrix = matrix.reindex([r for r in row_order if r in matrix.index])
    if col_order is not None:
        matrix = matrix.reindex(columns=[c for c in col_order if c in matrix.columns])
    return matrix


def draw_matrix_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    fmt: str,
    cmap,
    higher_better: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = pd.Series(values.ravel()).dropna()
    if finite.empty:
        return
    if vmin is None:
        vmin = float(finite.min())
    if vmax is None:
        vmax = float(finite.max())
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left", pad=2.0, fontweight="bold", fontsize=8.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([format_group(v) for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(v) for v in matrix.index])
    norm = lambda x: (float(x) - vmin) / (vmax - vmin)
    for row_idx, row in enumerate(matrix.index):
        for col_idx, col in enumerate(matrix.columns):
            value = matrix.loc[row, col]
            if pd.isna(value):
                continue
            nval = norm(value)
            dark_cell = nval > 0.56 if higher_better else nval < 0.44
            ax.text(
                col_idx,
                row_idx,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=6.9,
                color="white" if dark_cell else "#111111",
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.75)
    ax.tick_params(length=0, pad=1.4, labelsize=7.1)


def main_matrix_overview(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
) -> None:
    df = df.dropna(subset=["N", "policy", "goodput_tok_s", "p99_tpot_ms", "slo_attainment", "mean_selected_k"])
    if df.empty:
        return
    ns = sorted(int(n) for n in df["N"].dropna().unique())
    policies = ["FixedK-1", "FixedK-7", "TETRIS-Proxy", "SpecSched"]
    base_good = matrix_from_pivot(df, index="policy", columns="N", values="goodput_tok_s", row_order=["NoSpec"], col_order=ns)
    base_p99 = matrix_from_pivot(df, index="policy", columns="N", values="p99_tpot_ms", row_order=["NoSpec"], col_order=ns)
    if base_good.empty or base_p99.empty:
        return
    good = matrix_from_pivot(df, index="policy", columns="N", values="goodput_tok_s", row_order=policies, col_order=ns)
    p99 = matrix_from_pivot(df, index="policy", columns="N", values="p99_tpot_ms", row_order=policies, col_order=ns)
    slo = matrix_from_pivot(
        df,
        index="policy",
        columns="N",
        values="slo_attainment",
        row_order=["NoSpec", "TETRIS-Proxy", "SpecSched"],
        col_order=ns,
    )
    selected_k = matrix_from_pivot(df, index="policy", columns="N", values="mean_selected_k", row_order=policies, col_order=ns)

    speedup = good.divide(base_good.loc["NoSpec"], axis=1)
    p99_reduction = (base_p99.loc["NoSpec"] - p99).divide(base_p99.loc["NoSpec"], axis=1) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.72))
    draw_matrix_heatmap(
        axes[0, 0],
        speedup,
        title="(a) Goodput speedup vs. NoSpec",
        fmt="{:.2f}x",
        cmap=TEAL_CMAP,
        higher_better=True,
        vmin=1.0,
        vmax=max(2.05, float(speedup.max().max())),
    )
    draw_matrix_heatmap(
        axes[0, 1],
        p99_reduction,
        title="(b) P99 TPOT reduction vs. NoSpec",
        fmt="{:.0f}%",
        cmap=BLUE_CMAP,
        higher_better=True,
        vmin=0.0,
        vmax=max(45.0, float(p99_reduction.max().max())),
    )
    draw_matrix_heatmap(
        axes[1, 0],
        slo,
        title="(c) SLO attainment",
        fmt="{:.2f}",
        cmap=BLUE_CMAP,
        higher_better=True,
        vmin=0.80,
        vmax=1.0,
    )
    draw_matrix_heatmap(
        axes[1, 1],
        selected_k,
        title="(d) Mean selected draft length",
        fmt="{:.1f}",
        cmap=AMBER_CMAP,
        higher_better=True,
        vmin=0.0,
        vmax=7.0,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("Concurrent requests (N)")
    fig.tight_layout(pad=0.35, w_pad=0.62, h_pad=0.58)
    save_prelaid_figure(fig, outdir, name)


def line_vs_budget(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    outdir: Path,
    name: str,
    policies: Sequence[str] | None = None,
    n_value: int | None = None,
    log_y: bool = False,
) -> None:
    df = df.dropna(subset=["configured_kv_blocks", "N", "policy", metric])
    if df.empty:
        return
    if n_value is None:
        n_value = int(df["N"].max())
    df = df[df["N"] == n_value].sort_values("configured_kv_blocks")
    if df.empty:
        return
    budgets = sorted(int(b) for b in df["configured_kv_blocks"].dropna().unique())
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for policy in ordered_policies(df, policies):
        sub = df[df["policy"] == policy].sort_values("configured_kv_blocks")
        if sub.empty:
            continue
        plot_policy_line(ax, sub["configured_kv_blocks"].tolist(), sub[metric].tolist(), policy)
    ax.set_xticks(budgets)
    ax.set_xlabel("KV block budget")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    style_axis(ax)
    ax.text(
        0.98,
        0.96,
        f"N={n_value}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "linewidth": 0.35, "pad": 1.2},
    )
    style_legend(ax, ncol=2)
    save_figure(fig, outdir, name)


def line_grid_vs_budget(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
    policies: Sequence[str] | None = None,
    n_value: int | None = None,
) -> None:
    metrics = [
        ("goodput_tok_s", "SLO goodput (tok/s)", "(a) Goodput", False),
        ("p99_tpot_ms", "P99 TPOT (ms/token)", "(b) Tail latency", False),
        ("reprefills_per_1k_tokens", "Re-prefills / 1K tok.", "(c) Re-prefill pressure", False),
        ("kv_evictions_per_1k_tokens", "Evictions / 1K tok.", "(d) Eviction pressure", False),
    ]
    df = df.dropna(subset=["configured_kv_blocks", "N", "policy"])
    if df.empty:
        return
    if n_value is None:
        n_value = int(df["N"].max())
    df = df[df["N"] == n_value].copy()
    if df.empty:
        return
    budgets = sorted(int(b) for b in df["configured_kv_blocks"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.65), sharex=True)
    handles = labels = None
    for ax, (metric, ylabel, title, log_y) in zip(axes.ravel(), metrics):
        metric_df = df.dropna(subset=[metric])
        for policy in ordered_policies(metric_df, policies):
            sub = metric_df[metric_df["policy"] == policy].sort_values("configured_kv_blocks")
            if sub.empty:
                continue
            plot_policy_line(ax, sub["configured_kv_blocks"].tolist(), sub[metric].tolist(), policy)
        ax.set_xticks(budgets)
        ax.set_title(title, loc="left", pad=2.0, fontweight="bold")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        style_axis(ax)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for ax in axes[-1, :]:
        ax.set_xlabel("KV block budget")
    axes[0, 1].text(
        0.98,
        0.95,
        f"N={n_value}",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "linewidth": 0.35, "pad": 1.2},
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(len(labels), 5),
            frameon=False,
            columnspacing=0.95,
            handlelength=1.65,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.45, w_pad=0.72, h_pad=0.58)
    save_prelaid_figure(fig, outdir, name)


def best_alt_by_budget(df: pd.DataFrame, metric: str, *, minimize: bool, exclude: str = "SpecSched") -> pd.DataFrame:
    alt = df[df["policy"] != exclude].copy()
    if alt.empty:
        return pd.DataFrame()
    idx = alt.groupby("configured_kv_blocks")[metric].idxmin() if minimize else alt.groupby("configured_kv_blocks")[metric].idxmax()
    return alt.loc[idx].sort_values("configured_kv_blocks")


def annotate_policy_codes(ax: plt.Axes, rows: pd.DataFrame, metric: str, *, y_offset: int = 7) -> None:
    code = {
        "SpecSched-vLLM": "vLLM",
        "SpecSched-LRU": "LRU",
        "SpecSched-Longest": "Long.",
        "SpecSched-H2O": "H2O",
    }
    for _, row in rows.iterrows():
        ax.annotate(
            code.get(str(row["policy"]), str(row["policy"]).replace("SpecSched-", "")),
            xy=(row["configured_kv_blocks"], row[metric]),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if y_offset >= 0 else "top",
            fontsize=6.1,
            color="#4B5563",
        )


def kv_clean_lines(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
    n_value: int | None = None,
) -> None:
    df = df.dropna(subset=["configured_kv_blocks", "N", "policy"])
    if df.empty:
        return
    if n_value is None:
        n_value = int(df["N"].max())
    df = df[df["N"] == n_value].copy()
    if df.empty:
        return
    budgets = sorted(int(b) for b in df["configured_kv_blocks"].dropna().unique())
    spec = df[df["policy"] == "SpecSched"].sort_values("configured_kv_blocks")
    lru = df[df["policy"] == "SpecSched-LRU"].sort_values("configured_kv_blocks")
    best_good = best_alt_by_budget(df, "goodput_tok_s", minimize=False)
    best_p99 = best_alt_by_budget(df, "p99_tpot_ms", minimize=True)
    best_refill = best_alt_by_budget(df, "reprefills_per_1k_tokens", minimize=True)
    best_evict = best_alt_by_budget(df, "kv_evictions_per_1k_tokens", minimize=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.28), sharex=False)

    spec_by_budget = spec.set_index("configured_kv_blocks")
    best_good_by_budget = df.groupby("configured_kv_blocks")["goodput_tok_s"].max()
    best_p99_by_budget = df.groupby("configured_kv_blocks")["p99_tpot_ms"].min()
    good_gap = (spec_by_budget["goodput_tok_s"] / best_good_by_budget - 1.0) * 100.0
    p99_overhead = (spec_by_budget["p99_tpot_ms"] / best_p99_by_budget - 1.0) * 100.0

    ax = axes[0]
    xs = list(range(len(budgets)))
    vals = [good_gap.loc[b] for b in budgets]
    ax.bar(xs, vals, width=0.62, color="#006D5B", edgecolor="#1A1A1A", linewidth=0.55)
    ax.axhline(0, color="#1A1A1A", linewidth=0.8)
    for x, val in zip(xs, vals):
        if pd.isna(val):
            continue
        ax.text(x, val - 0.16 if val < 0 else val + 0.08, f"{val:.1f}%", ha="center", va="top" if val < 0 else "bottom", fontsize=6.7)
    ax.set_title("(a) Goodput gap to best", loc="left", pad=2.0, fontweight="bold")
    ax.set_ylabel("SpecSched gap (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in budgets])
    finite_vals = [float(v) for v in vals if pd.notna(v)]
    if finite_vals:
        ax.set_ylim(min(finite_vals) - 0.55, max(0.35, max(finite_vals) + 0.35))
    style_axis(ax)

    ax = axes[1]
    vals = [p99_overhead.loc[b] for b in budgets]
    ax.bar(xs, vals, width=0.62, color="#6F86A8", edgecolor="#1A1A1A", linewidth=0.55)
    for x, val in zip(xs, vals):
        if pd.isna(val):
            continue
        ax.text(x, val + 0.75, f"+{val:.0f}%", ha="center", va="bottom", fontsize=6.7)
    ax.set_title("(b) Tail-latency overhead", loc="left", pad=2.0, fontweight="bold")
    ax.set_ylabel("SpecSched overhead (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in budgets])
    finite_vals = [float(v) for v in vals if pd.notna(v)]
    if finite_vals:
        ax.set_ylim(0, max(finite_vals) * 1.22)
    style_axis(ax)

    ax = axes[2]
    metric = "reprefills_per_1k_tokens"
    plot_series_line(
        ax,
        spec["configured_kv_blocks"].tolist(),
        spec[metric].tolist(),
        "SpecSched",
        color="#006D5B",
        marker="*",
        linestyle="-",
        focus=True,
    )
    if not lru.empty:
        plot_series_line(
            ax,
            lru["configured_kv_blocks"].tolist(),
            lru[metric].tolist(),
            "LRU",
            color="#B45F06",
            marker=">",
            linestyle=(0, (1.4, 1.2)),
            focus=False,
            alpha=0.86,
        )
    if not best_refill.empty:
        plot_series_line(
            ax,
            best_refill["configured_kv_blocks"].tolist(),
            best_refill[metric].tolist(),
            "Best alternative",
            color="#4B5563",
            marker="o",
            linestyle=(0, (4.5, 1.4)),
            focus=False,
            alpha=0.78,
        )
    ax.set_title("(c) Cache pressure", loc="left", pad=2.0, fontweight="bold")
    ax.set_ylabel("Re-prefills / 1K tok.")
    ax.set_xticks(budgets)
    style_axis(ax)
    for ax in axes:
        ax.set_xlabel("KV block budget")
    handles, labels = axes[2].get_legend_handles_labels()
    legend = axes[2].legend(
        handles,
        labels,
        title=f"N={n_value}",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.22,
        handlelength=1.35,
        handletextpad=0.42,
        fontsize=6.8,
        title_fontsize=7.0,
    )
    legend.get_frame().set_edgecolor("#D0D0D0")
    legend.get_frame().set_linewidth(0.35)
    fig.tight_layout(pad=0.35, w_pad=0.75)
    save_prelaid_figure(fig, outdir, name)


def kv_matrix_overview(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
    policies: Sequence[str] | None = None,
    n_value: int | None = None,
) -> None:
    df = df.dropna(subset=["configured_kv_blocks", "N", "policy"])
    if df.empty:
        return
    if n_value is None:
        n_value = int(df["N"].max())
    df = df[df["N"] == n_value].copy()
    if df.empty:
        return
    budgets = sorted(int(b) for b in df["configured_kv_blocks"].dropna().unique())
    policies = ordered_policies(df, policies)
    good = matrix_from_pivot(df, index="policy", columns="configured_kv_blocks", values="goodput_tok_s", row_order=policies, col_order=budgets)
    p99 = matrix_from_pivot(df, index="policy", columns="configured_kv_blocks", values="p99_tpot_ms", row_order=policies, col_order=budgets)
    refill = matrix_from_pivot(df, index="policy", columns="configured_kv_blocks", values="reprefills_per_1k_tokens", row_order=policies, col_order=budgets)
    evict = matrix_from_pivot(df, index="policy", columns="configured_kv_blocks", values="kv_evictions_per_1k_tokens", row_order=policies, col_order=budgets)
    if good.empty or p99.empty:
        return
    good_pct_best = good.divide(good.max(axis=0), axis=1) * 100.0
    p99_over_best = (p99.divide(p99.min(axis=0), axis=1) - 1.0) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.85))
    draw_matrix_heatmap(
        axes[0, 0],
        good_pct_best,
        title="(a) Goodput (% of per-budget best)",
        fmt="{:.1f}",
        cmap=TEAL_CMAP,
        higher_better=True,
        vmin=max(80.0, float(good_pct_best.min().min()) - 1.0),
        vmax=100.0,
    )
    draw_matrix_heatmap(
        axes[0, 1],
        p99_over_best,
        title="(b) P99 overhead vs. best",
        fmt="+{:.0f}%",
        cmap=BLUE_CMAP.reversed(),
        higher_better=False,
        vmin=0.0,
        vmax=max(30.0, float(p99_over_best.max().max())),
    )
    draw_matrix_heatmap(
        axes[1, 0],
        refill,
        title="(c) Re-prefills / 1K tokens",
        fmt="{:.1f}",
        cmap=AMBER_CMAP.reversed(),
        higher_better=False,
        vmin=0.0,
        vmax=max(15.0, float(refill.max().max())),
    )
    draw_matrix_heatmap(
        axes[1, 1],
        evict,
        title="(d) Evictions / 1K tokens",
        fmt="{:.1f}",
        cmap=AMBER_CMAP.reversed(),
        higher_better=False,
        vmin=0.0,
        vmax=max(15.0, float(evict.max().max())),
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("KV block budget")
    axes[0, 1].text(
        0.98,
        1.08,
        f"N={n_value}",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.1,
    )
    fig.tight_layout(pad=0.35, w_pad=0.62, h_pad=0.62)
    save_prelaid_figure(fig, outdir, name)


def grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    ylabel: str,
    outdir: Path,
    name: str,
    policies: Sequence[str] | None = None,
) -> None:
    df = df.dropna(subset=[group_col, "policy", metric])
    if df.empty:
        return
    groups = sorted(df[group_col].dropna().unique())
    policies = ordered_policies(df, policies)
    if not policies:
        return
    width = min(0.8 / max(len(policies), 1), 0.28)
    x = list(range(len(groups)))
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for idx, policy in enumerate(policies):
        vals = []
        for group in groups:
            sub = df[(df[group_col] == group) & (df["policy"] == policy)]
            vals.append(float(sub[metric].iloc[0]) if not sub.empty else float("nan"))
        offset = (idx - (len(policies) - 1) / 2) * width
        ax.bar(
            [v + offset for v in x],
            vals,
            width=width,
            label=policy,
            color=COLORS.get(policy, "#777777"),
            edgecolor="#202020",
            linewidth=0.45,
            hatch=HATCHES.get(policy, ""),
            zorder=3 if policy == "SpecSched" else 2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([format_group(g) for g in groups])
    ax.set_xlabel(format_label(group_col))
    ax.set_ylabel(ylabel)
    style_axis(ax)
    if name.startswith("ablation_"):
        style_legend(
            ax,
            ncol=2,
            handlelength=0.95,
            handletextpad=0.34,
            columnspacing=0.72,
            labelspacing=0.16,
            fontsize=6.25,
        )
    else:
        style_legend(ax, ncol=min(len(policies), 4), handlelength=1.15)
    save_figure(fig, outdir, name)


def combine_slo_rows(slo_rows: pd.DataFrame, main_rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not slo_rows.empty:
        frames.append(slo_rows)
    if not main_rows.empty:
        frames.append(
            main_rows[
                main_rows["policy"].isin(SLO_POLICIES)
                & main_rows["N"].isin([16, 32, 64])
                & main_rows["configured_beta"].notna()
            ]
        )
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["configured_beta", "N", "policy"], keep="last")
    return combined


def relative_matrix(df: pd.DataFrame, metric: str, *, mode: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="N", columns=["configured_beta", "policy"], values=metric, aggfunc="first")
    betas = sorted(df["configured_beta"].dropna().unique())
    ns = sorted(df["N"].dropna().unique())
    matrix = pd.DataFrame(index=ns, columns=betas, dtype=float)
    for beta in betas:
        for n_value in ns:
            try:
                spec = float(pivot.loc[n_value, (beta, "SpecSched")])
                base = float(pivot.loc[n_value, (beta, "IndepOpt")])
            except Exception:
                continue
            if mode == "gain":
                matrix.loc[n_value, beta] = (spec / base - 1.0) * 100.0
            elif mode == "reduction":
                matrix.loc[n_value, beta] = (base - spec) / base * 100.0
            elif mode == "spec":
                matrix.loc[n_value, beta] = spec
    return matrix


def draw_heatmap(ax: plt.Axes, matrix: pd.DataFrame, *, title: str, fmt: str, cmap) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = pd.Series(values.ravel()).dropna()
    if finite.empty:
        return
    vmin = max(0.0, float(finite.min()) * 0.85)
    vmax = float(finite.max()) * 1.08
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left", pad=2.0, fontweight="bold", fontsize=7.7)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([format_group(v) for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(int(v)) for v in matrix.index])
    ax.set_xlabel(r"SLO multiplier $\beta$")
    ax.set_ylabel("N")
    midpoint = vmin + 0.58 * (vmax - vmin)
    for row_idx, n_value in enumerate(matrix.index):
        for col_idx, beta in enumerate(matrix.columns):
            value = matrix.loc[n_value, beta]
            if pd.isna(value):
                continue
            color = "white" if value >= midpoint else "#111111"
            ax.text(col_idx, row_idx, fmt.format(value), ha="center", va="center", fontsize=7.4, color=color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.85)
    ax.tick_params(length=0, pad=2.0)
    return image


def slo_sensitivity_overview(slo_rows: pd.DataFrame, main_rows: pd.DataFrame, outdir: Path) -> None:
    df = combine_slo_rows(slo_rows, main_rows)
    df = filter_paper_policies(df, SLO_POLICIES)
    if df.empty:
        return
    goodput_gain = relative_matrix(df, "goodput_tok_s", mode="gain")
    p99_reduction = relative_matrix(df, "p99_tpot_ms", mode="reduction")
    if goodput_gain.empty or p99_reduction.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 1.68))
    draw_heatmap(axes[0], goodput_gain, title="(a) Goodput gain", fmt="{:.0f}%", cmap=TEAL_CMAP)
    draw_heatmap(axes[1], p99_reduction, title="(b) P99 reduction", fmt="{:.0f}%", cmap=BLUE_CMAP)
    fig.tight_layout(pad=0.25, w_pad=0.48)
    save_prelaid_figure(fig, outdir, "slo_sensitivity_overview")

    fig, ax = plt.subplots(figsize=(1.75, 1.72))
    draw_heatmap(ax, goodput_gain, title="Goodput gain vs. IndepOpt", fmt="{:.0f}%", cmap=TEAL_CMAP)
    fig.tight_layout(pad=0.35)
    save_prelaid_figure(fig, outdir, "slo_goodput_by_beta")

    attain = relative_matrix(df, "slo_attainment", mode="spec")
    if not attain.empty:
        fig, ax = plt.subplots(figsize=(1.75, 1.72))
        draw_heatmap(ax, attain, title="SpecSched SLO attainment", fmt="{:.2f}", cmap=BLUE_CMAP)
        fig.tight_layout(pad=0.35)
        save_prelaid_figure(fig, outdir, "slo_attainment_by_beta")


def parse_overhead_kmax(row: pd.Series) -> int:
    tag = str(row.get("tag", ""))
    if tag.startswith("overhead_k"):
        return int(tag[len("overhead_k"):])
    return int(float(row.get("x_kmax", 0)))


def plot_scheduler_overhead(outdir: Path) -> None:
    if not OVERHEAD_SUMMARY.exists():
        return
    df = normalize(pd.read_csv(OVERHEAD_SUMMARY))
    if df.empty or "policy" not in df.columns:
        return
    df = df[df["policy"] == "SpecSched"].copy()
    needed = [
        "N",
        "scheduler_bookkeeping_mean_us",
        "scheduler_policy_mean_us",
        "scheduler_kv_commit_mean_us",
        "scheduler_cleanup_mean_us",
        "scheduler_cpu_frac",
    ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        return
    df["kmax_plot"] = df.apply(parse_overhead_kmax, axis=1)
    df = df.dropna(subset=needed + ["kmax_plot"]).sort_values(["kmax_plot", "N"])
    if df.empty:
        return

    preferred_k = 5 if 5 in set(df["kmax_plot"]) else int(df["kmax_plot"].median())
    main = df[df["kmax_plot"] == preferred_k].sort_values("N")
    if main.empty:
        return

    phase_cols = [
        ("scheduler_policy_mean_us", "Policy", "#006D5B"),
        ("scheduler_cleanup_mean_us", "Cleanup", "#1F4E79"),
        ("scheduler_bookkeeping_mean_us", "Bookkeeping", "#9C4F00"),
        ("scheduler_kv_commit_mean_us", "KV commit", "#6B7280"),
    ]
    xs = list(range(len(main)))
    labels = [str(int(v)) for v in main["N"]]

    fig = plt.figure(figsize=(7.15, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    bottom = [0.0] * len(main)
    for col, label, color in phase_cols:
        vals = [float(v) for v in main[col]]
        ax.bar(
            xs,
            vals,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="#161616",
            linewidth=0.45,
            width=0.62,
            zorder=3,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, total in zip(xs, main["scheduler_cpu_mean_us"]):
        ax.text(x, float(total) + 7.0, f"{float(total):.0f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_title(f"(a) Phase breakdown at $k_{{\\max}}={preferred_k}$", loc="left", pad=2.0, fontweight="bold")
    ax.set_xlabel("Concurrent requests (N)")
    ax.set_ylabel("CPU scheduler overhead ($\\mu$s/epoch)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(bottom) * 1.24)
    style_axis(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#CFCFCF",
        borderpad=0.25,
        labelspacing=0.18,
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.72,
        fontsize=6.55,
    )
    legend.get_frame().set_linewidth(0.35)

    heat = df.pivot_table(index="kmax_plot", columns="N", values="scheduler_cpu_frac", aggfunc="first").sort_index()
    heat = heat.reindex(columns=sorted(heat.columns))
    ax = fig.add_subplot(gs[0, 1])
    values = heat.to_numpy(dtype=float) * 100.0
    vmax = max(0.30, float(pd.Series(values.ravel()).max()))
    ax.imshow(values, cmap=TEAL_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title("(b) Wall-clock fraction", loc="left", pad=2.0, fontweight="bold")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$k_{\max}$")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([str(int(v)) for v in heat.columns])
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([str(int(v)) for v in heat.index])
    midpoint = 0.55 * vmax
    for r, k_value in enumerate(heat.index):
        for c, n_value in enumerate(heat.columns):
            value = values[r, c]
            ax.text(
                c,
                r,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if value >= midpoint else "#111111",
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.85)
    ax.tick_params(length=0, pad=2.0)
    fig.tight_layout(pad=0.35)
    save_prelaid_figure(fig, outdir, "scheduler_overhead_stack")


def format_group(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_label(col: str) -> str:
    if col == "configured_beta":
        return "SLO multiplier beta"
    if col == "policy":
        return "Policy"
    if col == "N":
        return "Concurrent requests (N)"
    return col.replace("_", " ")


def write_table(df: pd.DataFrame, outdir: Path, name: str, columns: Iterable[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    columns = [c for c in columns if c in df.columns]
    table = df.loc[:, columns].copy()
    table.to_csv(outdir / f"{name}.csv", index=False)
    write_latex_table(table, outdir / f"{name}.tex")


def latex_escape(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    cols = list(table.columns)
    align = "l" * len(cols)
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def add_speedups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "N" not in df.columns or "policy" not in df.columns:
        return df
    baselines = df[df["policy"] == "NoSpec"][["N", "throughput_tok_s", "goodput_tok_s"]].rename(
        columns={
            "throughput_tok_s": "nospec_throughput_tok_s",
            "goodput_tok_s": "nospec_goodput_tok_s",
        }
    )
    out = df.merge(baselines, on="N", how="left")
    out["throughput_speedup_vs_nospec"] = out["throughput_tok_s"] / out["nospec_throughput_tok_s"]
    out["goodput_speedup_vs_nospec"] = out["goodput_tok_s"] / out["nospec_goodput_tok_s"]
    return out.drop(columns=["nospec_throughput_tok_s", "nospec_goodput_tok_s"], errors="ignore")


def write_tables(
    all_rows: pd.DataFrame,
    main_rows: pd.DataFrame,
    kv_rows: pd.DataFrame,
    slo_rows: pd.DataFrame,
    ablation_rows: pd.DataFrame,
    outdir: Path,
) -> None:
    table_cols = [
        "tag",
        "kind",
        "policy",
        "N",
        "configured_beta",
        "configured_kv_blocks",
        "throughput_tok_s",
        "goodput_tok_s",
        "slo_attainment",
        "p99_tpot_ms",
        "acceptance_rate",
        "target_forwards_per_token",
        "reprefills_per_1k_tokens",
        "kv_evictions_per_1k_tokens",
        "x_peak_gpu_mb",
    ]
    all_rows = filter_paper_policies(all_rows)
    main_rows = filter_paper_policies(main_rows, MAIN_POLICIES)
    kv_rows = filter_paper_policies(kv_rows, KV_POLICIES)
    slo_rows = filter_paper_policies(slo_rows, SLO_POLICIES)
    ablation_rows = filter_paper_policies(ablation_rows, ABLATION_POLICIES)
    write_table(all_rows.sort_values(["kind", "tag", "N", "policy"]), outdir, "all_results", table_cols)

    if not main_rows.empty:
        main_table = add_speedups(main_rows)
        for n_value in sorted(main_table["N"].dropna().unique()):
            selected = main_table[main_table["N"] == n_value].sort_values("policy")
            write_table(
                selected,
                outdir,
                f"main_n{int(n_value)}",
                table_cols + ["throughput_speedup_vs_nospec", "goodput_speedup_vs_nospec"],
            )

    if not kv_rows.empty:
        n_value = int(kv_rows["N"].max())
        selected = kv_rows[kv_rows["N"] == n_value].sort_values(["configured_kv_blocks", "policy"])
        write_table(selected, outdir, f"kv_sweep_n{n_value}", table_cols)

    if not slo_rows.empty:
        selected = slo_rows.sort_values(["configured_beta", "N", "policy"])
        write_table(selected, outdir, "slo_sensitivity", table_cols)

    if not ablation_rows.empty:
        selected = ablation_rows.sort_values(["N", "policy"])
        write_table(selected, outdir, "ablation", table_cols)


def plot_all(
    main_rows: pd.DataFrame,
    kv_rows: pd.DataFrame,
    slo_rows: pd.DataFrame,
    ablation_rows: pd.DataFrame,
    outdir: Path,
) -> None:
    full_main_rows = main_rows.copy()
    main_rows = filter_paper_policies(main_rows, MAIN_POLICIES)
    kv_rows = filter_paper_policies(kv_rows, KV_POLICIES)
    slo_rows = filter_paper_policies(slo_rows, SLO_POLICIES)
    ablation_rows = filter_paper_policies(ablation_rows, ABLATION_POLICIES)

    line_vs_n(main_rows, "throughput_tok_s", "Throughput (tokens/s)", outdir, "main_throughput_vs_n", policies=MAIN_POLICIES)
    line_vs_n(main_rows, "goodput_tok_s", "SLO goodput (tokens/s)", outdir, "main_goodput_vs_n", policies=MAIN_POLICIES)
    line_vs_n(main_rows, "slo_attainment", "SLO attainment", outdir, "main_slo_attainment_vs_n", policies=MAIN_POLICIES)
    line_vs_n(main_rows, "p99_tpot_ms", "P99 TPOT (ms/token)", outdir, "main_p99_tpot_vs_n", policies=MAIN_POLICIES, log_y=True)
    line_vs_n(main_rows, "x_peak_gpu_mb", "Peak GPU memory (MB)", outdir, "main_peak_gpu_memory_vs_n", policies=MAIN_POLICIES)
    main_clean_lines(full_main_rows, outdir, "main_scheduler_overview")

    line_vs_budget(kv_rows, "goodput_tok_s", "SLO goodput (tokens/s)", outdir, "kv_goodput_vs_budget", policies=KV_POLICIES)
    line_vs_budget(kv_rows, "p99_tpot_ms", "P99 TPOT (ms/token)", outdir, "kv_p99_tpot_vs_budget", policies=KV_POLICIES, log_y=True)
    line_vs_budget(kv_rows, "reprefills_per_1k_tokens", "Re-prefills / 1K tokens", outdir, "kv_refill_vs_budget", policies=KV_POLICIES)
    line_vs_budget(kv_rows, "kv_evictions_per_1k_tokens", "KV evictions / 1K tokens", outdir, "kv_evictions_vs_budget", policies=KV_POLICIES)
    kv_clean_lines(kv_rows, outdir, "kv_sweep_overview")

    slo_sensitivity_overview(slo_rows, full_main_rows, outdir)
    grouped_bar(ablation_rows, "N", "throughput_tok_s", "Throughput (tokens/s)", outdir, "ablation_throughput", policies=ABLATION_POLICIES)
    grouped_bar(ablation_rows, "N", "p99_tpot_ms", "P99 TPOT (ms/token)", outdir, "ablation_p99_tpot", policies=ABLATION_POLICIES)
    plot_scheduler_overhead(outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=None, help="Experiment results root containing processed/*.csv.")
    parser.add_argument("--summary", default=None, help="Processed summary CSV. Overrides --results-root.")
    parser.add_argument("--figures-dir", default="paper_assets/figures")
    parser.add_argument("--tables-dir", default="paper_assets/tables")
    args = parser.parse_args()

    configure_matplotlib()
    processed_dir = processed_dir_from_results_root(args.results_root)
    all_rows, input_path = load_summary(args.summary, processed_dir)
    main_rows = load_split("main", all_rows, processed_dir)
    kv_rows = load_split("kv_sweep", all_rows, processed_dir)
    slo_rows = load_split("slo_sensitivity", all_rows, processed_dir)
    ablation_rows = load_split("ablation", all_rows, processed_dir)

    figures_dir = resolve(args.figures_dir)
    tables_dir = resolve(args.tables_dir)
    plot_all(main_rows, kv_rows, slo_rows, ablation_rows, figures_dir)
    write_tables(all_rows, main_rows, kv_rows, slo_rows, ablation_rows, tables_dir)

    main_figures_dir = ROOT / "main/figures"
    if main_figures_dir.resolve() != figures_dir.resolve():
        for name in ("scheduler_overhead_stack.pdf", "scheduler_overhead_stack.png"):
            src = figures_dir / name
            if src.exists():
                main_figures_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, main_figures_dir / name)

    manifest = {
        "input_summary": str(input_path),
        "processed_dir": str(processed_dir),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "rows": int(len(all_rows)),
        "generated_figures": sorted(p.name for p in figures_dir.glob("*") if p.suffix in {".pdf", ".png"}),
        "generated_tables": sorted(p.name for p in tables_dir.glob("*") if p.suffix in {".csv", ".tex"}),
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

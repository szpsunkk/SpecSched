#!/usr/bin/env python3
"""Generate publication-style alternative figures from saved results.

The script does not rerun experiments and does not overwrite manuscript figures.
Outputs go to paper_assets/figures_scientific and main/figures_scientific.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

METHOD = {
    "SpecSched": (PALETTE["blue_main"], "*", "-", 2.8, 8.0, 1.0),
    "NoSpec": (PALETTE["neutral_dark"], "o", (0, (3.2, 1.5)), 1.8, 5.5, 0.82),
    "BestFixed": (PALETTE["red_strong"], "D", "-", 2.0, 5.8, 0.86),
    "TETRIS-Proxy": (PALETTE["teal"], "X", (0, (1.4, 1.2)), 1.9, 5.8, 0.86),
    "IndepOpt": (PALETTE["violet"], "P", (0, (4.5, 1.4)), 1.8, 5.5, 0.82),
    "SpecServe-Proxy": (PALETTE["green_3"], "s", (0, (2.2, 1.3)), 1.8, 5.4, 0.82),
    "EAGLE-2-Proxy": (PALETTE["red_2"], "^", (0, (5.0, 1.4)), 1.8, 5.4, 0.82),
    "LRU": (PALETTE["red_2"], ">", (0, (3.0, 1.4)), 1.8, 5.4, 0.84),
    "Longest": (PALETTE["violet"], "8", (0, (1.4, 1.2)), 1.8, 5.4, 0.84),
    "H2O": (PALETTE["green_3"], "p", (0, (4.5, 1.4)), 1.8, 5.4, 0.84),
    "vLLM": (PALETTE["teal"], "d", (0, (2.2, 1.3)), 1.8, 5.4, 0.84),
}

HATCHES = {
    "SpecSched": "",
    "SpecSched-NoBatch": "//",
    "SpecSched-NoEvict": "\\\\",
    "SpecSched-NoSLO": "..",
    "NoSpec": "",
    "SpecSched-Safe": "xx",
    "TETRIS-Proxy": "**",
    "FixedK-1": "//",
    "FixedK-3": "\\\\",
}

TEAL_CMAP = LinearSegmentedColormap.from_list(
    "specsched_teal", ["#F4FAF8", PALETTE["green_2"], PALETTE["teal"], PALETTE["blue_main"]]
)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "specsched_blue", ["#F5F8FC", "#C9DDF0", PALETTE["blue_secondary"], PALETTE["blue_main"]]
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "axes.linewidth": 2.1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "hatch.linewidth": 0.8,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {"policy", "baseline", "focus_policy", "seeds", "baseline_labels", "arrival_model"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def save(fig: plt.Figure, outdir: Path, name: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kwargs in {
        "pdf": {},
        "png": {"dpi": 600},
        "svg": {},
    }.items():
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def style_axis(ax: plt.Axes, *, grid_x: bool = False) -> None:
    ax.grid(True, axis="y", linestyle=(0, (1.0, 2.0)), linewidth=0.8, color="#CFCFCF", alpha=0.85)
    if grid_x:
        ax.grid(True, axis="x", linestyle=(0, (1.0, 2.0)), linewidth=0.55, color="#D9D9D9", alpha=0.55)
    ax.tick_params(direction="out", length=4.0, width=1.2, pad=3.0)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)


def style_for(label: str):
    return METHOD.get(label, (PALETTE["neutral_dark"], "o", "-", 1.9, 5.5, 0.85))


def plot_line(ax: plt.Axes, x, y, yerr, label: str) -> None:
    color, marker, linestyle, linewidth, markersize, alpha = style_for(label)
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        alpha=alpha,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=5 if label == "SpecSched" else 3,
    )
    if yerr is not None and np.isfinite(yerr).any() and float(np.nanmax(yerr)) > 0:
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12 if label == "SpecSched" else 0.08, linewidth=0)


def best_fixed_from_main(summary: pd.DataFrame) -> pd.DataFrame:
    fixed = summary[summary["policy"].isin(["FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7"])].copy()
    idx = fixed.groupby("N")["goodput_tok_s_mean"].idxmax()
    best = fixed.loc[idx].copy()
    best["policy"] = "BestFixed"
    return best


def load_transfer_plot_rows() -> tuple[pd.DataFrame, str]:
    multi_summary = ROOT / "results/tpds_multiseed_transfer_gpu/processed/main_multiseed_summary.csv"
    multi_lifts = ROOT / "results/tpds_multiseed_transfer_gpu/processed/main_multiseed_paired_lifts.csv"
    order = ["Workload shift", "Bursty arrivals", "Tiered SLOs"]
    tag_to_scenario = {
        "paper3090_transfer_shift": "Workload shift",
        "paper3090_transfer_bursty": "Bursty arrivals",
        "paper3090_transfer_tiered": "Tiered SLOs",
    }
    if multi_summary.exists() and multi_lifts.exists():
        summary = read_csv(multi_summary)
        lifts = read_csv(multi_lifts)
        fixed = summary[summary["policy"].isin(["FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7"])].copy()
        if not fixed.empty and "goodput_tok_s_mean" in fixed:
            idx = fixed.groupby("tag")["goodput_tok_s_mean"].idxmax()
            best = fixed.loc[idx].set_index("tag")
            spec = summary[summary["policy"].eq("SpecSched")].set_index("tag")
            paired = lifts[lifts["baseline"].eq("BestFixed")].set_index("tag")
            rows = []
            for tag, scenario in tag_to_scenario.items():
                if tag not in best.index or tag not in spec.index or tag not in paired.index:
                    continue
                rows.append(
                    {
                        "scenario": scenario,
                        "bestfixed_goodput_tok_s": float(best.loc[tag, "goodput_tok_s_mean"]),
                        "specsched_goodput_tok_s": float(spec.loc[tag, "goodput_tok_s_mean"]),
                        "specsched_lift_vs_bestfixed_pct": float(paired.loc[tag, "goodput_lift_pct_mean"]),
                        "specsched_lift_ci95_half_width": float(paired.loc[tag, "goodput_lift_pct_ci95_half_width"]),
                        "bestfixed_p99_tpot_ms": float(best.loc[tag, "p99_tpot_ms_mean"]),
                        "specsched_p99_tpot_ms": float(spec.loc[tag, "p99_tpot_ms_mean"]),
                        "source_label": "three-seed measured replay",
                    }
                )
            df = pd.DataFrame(rows)
            if set(order).issubset(set(df.get("scenario", []))):
                return df, "multi-seed"

    seed7 = ROOT / "results/transfer_gpu/processed/transfer_gpu_bestfixed_summary.csv"
    if not seed7.exists():
        return pd.DataFrame(), "missing"
    df = read_csv(seed7)
    if "specsched_lift_ci95_half_width" not in df:
        df["specsched_lift_ci95_half_width"] = np.nan
    df["source_label"] = "seed-7 measured replay"
    return df, "seed7"


def plot_main_scheduler(outdir: Path) -> None:
    summary = read_csv(ROOT / "results/tpds_multiseed_main/processed/main_multiseed_summary.csv")
    rows = pd.concat(
        [
            summary[summary["policy"].isin(["NoSpec", "TETRIS-Proxy", "SpecSched"])],
            best_fixed_from_main(summary),
        ],
        ignore_index=True,
    )
    rows = rows[rows["N"].isin([16, 32, 64])].copy()
    order = ["NoSpec", "BestFixed", "TETRIS-Proxy", "SpecSched"]
    panels = [
        ("goodput_tok_s", "SLO-goodput (tok/s)", False),
        ("p99_tpot_ms", "P99 TPOT (ms/token)", False),
        ("slo_attainment", "SLO attainment", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.7]})
    for ax, (metric, ylabel, log_y) in zip(axes[:3], panels):
        for label in order:
            sub = rows[rows["policy"] == label].sort_values("N")
            if sub.empty:
                continue
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            yerr_col = f"{metric}_std"
            yerr = sub[yerr_col].to_numpy(dtype=float) if yerr_col in sub else None
            plot_line(ax, sub["N"].to_numpy(dtype=float), y, yerr, label)
        ax.set_xlabel("Concurrent users N")
        ax.set_ylabel(ylabel)
        ax.set_xticks([16, 32, 64])
        if log_y:
            ax.set_yscale("log")
        style_axis(ax, grid_x=True)
    axes[0].set_title("(a) Goodput", loc="left", fontweight="bold")
    axes[1].set_title("(b) Tail latency", loc="left", fontweight="bold")
    axes[2].set_title("(c) SLO attainment", loc="left", fontweight="bold")
    axes[3].set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(handles, labels, loc="center left", fontsize=15, handlelength=2.2, labelspacing=1.0)
    fig.tight_layout(pad=1.0, w_pad=1.7)
    save(fig, outdir, "main_scheduler_overview_scientific")


def plot_slo_sensitivity(outdir: Path) -> None:
    rows = pd.concat(
        [
            read_csv(ROOT / "results/local_v3_kcap3/processed/scheduler_slo_sensitivity_summary.csv"),
            read_csv(ROOT / "results/local_v3_kcap3/processed/scheduler_main_summary.csv"),
        ],
        ignore_index=True,
    )
    tag_by_beta = {
        1.25: "paper3090_slo_beta125",
        1.5: "paper3090_main_beta15",
        2.0: "paper3090_slo_beta20",
    }
    ns = [16, 32, 64]
    betas = [1.25, 1.5, 2.0]
    mat = pd.DataFrame(index=ns, columns=betas, dtype=float)
    for n in ns:
        for beta in betas:
            tag = tag_by_beta[beta]
            spec = rows[(rows["tag"].eq(tag)) & (rows["N"].eq(n)) & (rows["policy"].eq("SpecSched"))]
            indep = rows[(rows["tag"].eq(tag)) & (rows["N"].eq(n)) & (rows["policy"].eq("IndepOpt"))]
            if spec.empty or indep.empty:
                continue
            mat.loc[n, beta] = (float(spec["goodput_tok_s"].iloc[0]) / float(indep["goodput_tok_s"].iloc[0]) - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(5.3, 3.35))
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin, vmax = 0.0, float(finite.max()) * 1.08
    image = ax.imshow(vals, cmap=TEAL_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title("SpecSched gain over IndepOpt", loc="left", fontweight="bold", fontsize=15)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    ax.set_xlabel("SLO multiplier beta")
    ax.set_ylabel("Concurrent users N")
    threshold = vmin + 0.58 * (vmax - vmin)
    for i, n in enumerate(ns):
        for j, beta in enumerate(betas):
            value = mat.loc[n, beta]
            if pd.isna(value):
                continue
            color = "white" if value >= threshold else "#111111"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=13, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Goodput gain (%)")
    fig.tight_layout(pad=1.0)
    save(fig, outdir, "slo_sensitivity_overview_scientific")


def policy_alias(policy: str) -> str:
    return {
        "SpecSched-LRU": "LRU",
        "SpecSched-Longest": "Longest",
        "SpecSched-H2O": "H2O",
        "SpecSched-vLLM": "vLLM",
    }.get(policy, policy)


def plot_kv_sweep(outdir: Path) -> None:
    df = read_csv(ROOT / "results/tpds_multiseed_kv_pressure/processed/main_multiseed_summary.csv")
    df = df[df["N"].eq(64)].copy()
    if "configured_kv_blocks" not in df.columns:
        df["configured_kv_blocks"] = df["kv_blocks_total"]
    df["alias"] = df["policy"].map(policy_alias)
    order = ["SpecSched", "vLLM", "LRU", "Longest", "H2O"]
    panels = [
        ("goodput_tok_s", "SLO-goodput (tok/s)"),
        ("p99_tpot_ms", "P99 TPOT (ms/token)"),
        ("reprefills_per_1k_tokens", "Re-prefills / 1K tokens"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.68]})
    for ax, (metric, ylabel) in zip(axes[:3], panels):
        for label in order:
            sub = df[df["alias"] == label].sort_values("configured_kv_blocks")
            if sub.empty:
                continue
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            yerr_col = f"{metric}_std"
            yerr = sub[yerr_col].to_numpy(dtype=float) if yerr_col in sub else None
            plot_line(ax, sub["configured_kv_blocks"].to_numpy(dtype=float), y, yerr, label)
        ax.set_xlabel("KV block budget")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["configured_kv_blocks"].dropna().unique()))
        style_axis(ax, grid_x=True)
    axes[0].set_title("(a) Goodput", loc="left", fontweight="bold")
    axes[1].set_title("(b) Tail latency", loc="left", fontweight="bold")
    axes[2].set_title("(c) Cache churn", loc="left", fontweight="bold")
    axes[3].set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(handles, labels, loc="center left", fontsize=15, handlelength=2.2, labelspacing=0.9)
    fig.tight_layout(pad=1.0, w_pad=1.65)
    save(fig, outdir, "kv_sweep_overview_scientific")


def plot_route_kv_stress(outdir: Path) -> None:
    df = read_csv(ROOT / "results/processed/route_kv_stress_summary.csv")
    route_order = ["local-only", "random", "kv-aware", "oracle"]
    route_labels = ["Local", "Random", "KV-aware", "Oracle"]
    route_colors = [PALETTE["neutral_dark"], PALETTE["red_strong"], PALETTE["teal"], PALETTE["blue_main"]]

    focus = df[(df["N"].eq(64)) & (df["kv_blocks"].eq(512)) & (df["route_delay_ms"].eq(5.0))].copy()
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.1))
    metrics = [
        ("goodput_tok_s_mean", "SLO-goodput (tok/s)", False),
        ("p99_tpot_ms_mean", "P99 TPOT (ms/token)", False),
        ("kv_utilization_mean", "KV utilization", False),
        ("kv_evictions_per_1k_tokens_mean", "Evictions / 1K tokens", False),
    ]
    x = np.arange(len(route_order))
    for ax, (metric, ylabel, log_y) in zip(axes, metrics):
        vals = []
        errs = []
        for mode in route_order:
            row = focus[focus["routing_mode"].eq(mode)]
            vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            std_col = metric.replace("_mean", "_std")
            errs.append(float(row[std_col].iloc[0]) if not row.empty and std_col in row else 0.0)
        bars = ax.bar(
            x,
            vals,
            yerr=errs if max(errs) > 0 else None,
            width=0.68,
            color=route_colors,
            edgecolor="black",
            linewidth=1.2,
            error_kw={"elinewidth": 1.0, "capsize": 3.0, "capthick": 1.0},
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            if not np.isfinite(value):
                continue
            label = f"{value:.2f}" if value < 1 else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(vals) * 0.025, label, ha="center", va="bottom", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(route_labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        style_axis(ax)
    axes[0].set_title("(a) Throughput", loc="left", fontweight="bold")
    axes[1].set_title("(b) Tail latency", loc="left", fontweight="bold")
    axes[2].set_title("(c) KV pressure", loc="left", fontweight="bold")
    axes[3].set_title("(d) Cache churn", loc="left", fontweight="bold")
    fig.tight_layout(pad=1.0, w_pad=1.3)
    save(fig, outdir, "route_kv_ablation_scientific")

    heat = df[(df["routing_mode"].eq("kv-aware")) & (df["route_delay_ms"].eq(5.0))].copy()
    ns = sorted(heat["N"].dropna().unique())
    kvs = sorted(heat["kv_blocks"].dropna().unique())
    mat = np.zeros((len(ns), len(kvs)), dtype=float)
    for i, n_value in enumerate(ns):
        for j, kv_blocks in enumerate(kvs):
            row = heat[(heat["N"].eq(n_value)) & (heat["kv_blocks"].eq(kv_blocks))]
            mat[i, j] = float(row["goodput_tok_s_mean"].iloc[0]) if not row.empty else np.nan

    fig, ax = plt.subplots(figsize=(5.2, 3.45))
    image = ax.imshow(mat, cmap=BLUE_CMAP, aspect="auto")
    ax.set_title("Joint stress: KV-aware routing", loc="left", fontweight="bold", fontsize=13)
    ax.set_xlabel("KV block budget")
    ax.set_ylabel("Concurrent users N")
    ax.set_xticks(range(len(kvs)))
    ax.set_xticklabels([str(int(v)) for v in kvs])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(int(v)) for v in ns])
    threshold = np.nanmin(mat) + 0.58 * (np.nanmax(mat) - np.nanmin(mat))
    for i, n_value in enumerate(ns):
        for j, kv_blocks in enumerate(kvs):
            value = mat[i, j]
            color = "white" if value >= threshold else "#111111"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=12, color=color)
    ax.tick_params(length=0)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.6)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("SLO-goodput (tok/s)")
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.yaxis.label.set_size(12)
    fig.tight_layout(pad=1.0)
    save(fig, outdir, "route_kv_joint_stress_scientific")


def plot_bestfixed_stress(outdir: Path) -> None:
    df = read_csv(ROOT / "results/processed/bestfixed_stress_summary.csv")
    df = df[df["N"].eq(64)].copy()
    scenarios = ["workload_shift", "bursty_arrival", "tiered_slo"]
    scenario_labels = ["Workload\nshift", "Bursty\narrivals", "Tiered\nSLOs"]
    methods = ["BestFixed-Global", "BestFixed-PhaseTuned", "SpecSched"]
    method_labels = ["BestFixed", "Phase-tuned", "SpecSched"]
    colors = [PALETTE["red_strong"], PALETTE["neutral_dark"], PALETTE["blue_main"]]
    hatches = ["//", "", ""]

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 0.72]})
    metrics = [
        ("goodput_tok_s_mean", "SLO-goodput (tok/s)", "(a) Goodput"),
        ("p99_tpot_ms_mean", "P99 TPOT (ms/token)", "(b) Tail latency"),
        ("recovery_time_s_mean", "Recovery time (s)", "(c) Adaptation"),
    ]
    x = np.arange(len(scenarios))
    width = 0.23
    offsets = np.linspace(-width, width, len(methods))
    for ax, (metric, ylabel, title) in zip(axes[:3], metrics):
        for offset, method, label, color, hatch in zip(offsets, methods, method_labels, colors, hatches):
            vals = []
            errs = []
            for scenario in scenarios:
                row = df[(df["scenario"].eq(scenario)) & (df["method"].eq(method))]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
                std_col = metric.replace("_mean", "_std")
                errs.append(float(row[std_col].iloc[0]) if not row.empty and std_col in row else 0.0)
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                yerr=errs if max(errs) > 0 else None,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=1.1,
                hatch=hatch,
                error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
                zorder=3,
            )
            if metric == "goodput_tok_s_mean" and method == "SpecSched":
                for bar, value in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        style_axis(ax)
    axes[3].set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(handles, labels, loc="center left", fontsize=14, handlelength=1.8, labelspacing=1.0)
    fig.tight_layout(pad=1.0, w_pad=1.4)
    save(fig, outdir, "bestfixed_stress_scientific")


def plot_transfer_gpu_measured(outdir: Path) -> None:
    df, source = load_transfer_plot_rows()
    if df.empty or "specsched_goodput_tok_s" not in df.columns:
        return
    df = df.dropna(subset=["bestfixed_goodput_tok_s", "specsched_goodput_tok_s"]).copy()
    if df.empty:
        return

    order = ["Workload shift", "Bursty arrivals", "Tiered SLOs"]
    present = set(str(value) for value in df["scenario"])
    if not set(order).issubset(present):
        print(
            "skip measured transfer figure: incomplete scenarios "
            f"{sorted(present)}; need {order}"
        )
        return
    df["scenario_order"] = df["scenario"].map({label: idx for idx, label in enumerate(order)})
    df = df.sort_values(["scenario_order", "scenario"])
    labels = [str(x).replace(" ", "\n") for x in df["scenario"]]
    x = np.arange(len(df))
    width = 0.34
    title_prefix = "Three-seed" if source == "multi-seed" else "Seed-7"

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    ax = axes[0]
    ax.bar(
        x - width / 2,
        df["bestfixed_goodput_tok_s"],
        width=width,
        label="BestFixed",
        color=PALETTE["red_strong"],
        edgecolor="black",
        linewidth=1.1,
        hatch="//",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        df["specsched_goodput_tok_s"],
        width=width,
        label="SpecSched",
        color=PALETTE["blue_main"],
        edgecolor="black",
        linewidth=1.1,
        zorder=3,
    )
    for xpos, value in zip(x + width / 2, df["specsched_goodput_tok_s"]):
        ax.text(xpos, value + max(df["specsched_goodput_tok_s"]) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SLO-goodput (tok/s)")
    ax.set_title(f"(a) {title_prefix} replay", loc="left", fontweight="bold")
    style_axis(ax)

    ax = axes[1]
    colors = [PALETTE["blue_main"] if value >= 0 else PALETTE["red_strong"] for value in df["specsched_lift_vs_bestfixed_pct"]]
    yerr = df["specsched_lift_ci95_half_width"].to_numpy(dtype=float) if "specsched_lift_ci95_half_width" in df else None
    if yerr is not None and not np.isfinite(yerr).any():
        yerr = None
    ax.bar(
        x,
        df["specsched_lift_vs_bestfixed_pct"],
        yerr=yerr,
        capsize=4 if yerr is not None else 0,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        zorder=3,
    )
    ax.axhline(0, color="#111111", linewidth=1.0)
    for xpos, value in zip(x, df["specsched_lift_vs_bestfixed_pct"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.8 if value >= 0 else -0.8
        ax.text(xpos, value + offset, f"{value:+.1f}%", ha="center", va=va, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Lift vs BestFixed (%)")
    ax.set_title("(b) Online-control gap", loc="left", fontweight="bold")
    style_axis(ax)

    ax = axes[2]
    ax.scatter(
        df["bestfixed_p99_tpot_ms"],
        df["specsched_p99_tpot_ms"],
        s=90,
        color=PALETTE["blue_main"],
        edgecolor="black",
        linewidth=1.0,
        zorder=4,
    )
    lo = min(df["bestfixed_p99_tpot_ms"].min(), df["specsched_p99_tpot_ms"].min()) * 0.96
    hi = max(df["bestfixed_p99_tpot_ms"].max(), df["specsched_p99_tpot_ms"].max()) * 1.04
    ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral_dark"], linestyle=(0, (3, 2)), linewidth=1.2)
    for _, row in df.iterrows():
        ax.text(row["bestfixed_p99_tpot_ms"], row["specsched_p99_tpot_ms"], f" {str(row['scenario']).split()[0]}", fontsize=10, va="center")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("BestFixed P99 TPOT")
    ax.set_ylabel("SpecSched P99 TPOT")
    ax.set_title("(c) Tail-latency tradeoff", loc="left", fontweight="bold")
    style_axis(ax, grid_x=True)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=13, frameon=False)
    fig.tight_layout(pad=1.0, w_pad=1.5)
    save(fig, outdir, "transfer_gpu_measured_scientific")


def plot_ablation(outdir: Path) -> None:
    df = read_csv(ROOT / "results/processed/ablation_summary.csv")
    df = df[df["N"].eq(64)].copy()
    order = ["SpecSched", "SpecSched-NoBatch", "SpecSched-NoEvict", "SpecSched-NoSLO"]
    labels = ["Full", "NoBatch", "NoEvict", "NoSLO"]
    colors = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_2"], PALETTE["violet"]]
    metrics = [("throughput_tok_s", "Throughput (tok/s)"), ("p99_tpot_ms", "P99 TPOT (ms/token)")]
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        vals = []
        for policy in order:
            sub = df[df["policy"] == policy]
            vals.append(float(sub[metric].iloc[0]) if not sub.empty else np.nan)
        x = np.arange(len(order))
        bars = ax.bar(
            x,
            vals,
            width=0.66,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
        for bar, policy in zip(bars, order):
            bar.set_hatch(HATCHES[policy])
        for bar, value in zip(bars, vals):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(vals) * 0.025,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=13,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Ablation at N=64: {ylabel}", loc="left", fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.18)
        style_axis(ax)
        fig.tight_layout(pad=1.0)
        name = "ablation_throughput_scientific" if metric == "throughput_tok_s" else "ablation_p99_tpot_scientific"
        save(fig, outdir, name)


def plot_edge(outdir: Path) -> None:
    df = read_csv(ROOT / "results/tpds_multiseed_edge_practical/processed/main_multiseed_summary.csv")
    policies = ["NoSpec", "TETRIS-Proxy", "SpecSched", "SpecSched-Safe"]
    df = df[df["policy"].isin(policies)].copy()
    colors = {
        "NoSpec": PALETTE["neutral_dark"],
        "TETRIS-Proxy": PALETTE["teal"],
        "SpecSched": PALETTE["red_strong"],
        "SpecSched-Safe": PALETTE["blue_main"],
    }
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 0.7]})
    metrics = [
        ("goodput_tok_s", "SLO-goodput (tok/s)", False),
        ("p99_tpot_ms", "P99 TPOT (ms/token)", True),
        ("slo_attainment", "SLO attainment", False),
    ]
    ns = sorted(df["N"].dropna().unique())
    x = np.arange(len(ns))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(policies))
    for ax, (metric, ylabel, log_y) in zip(axes[:3], metrics):
        for offset, policy in zip(offsets, policies):
            sub = df[df["policy"].eq(policy)].set_index("N")
            vals = [float(sub.loc[n, f"{metric}_mean"]) if n in sub.index else np.nan for n in ns]
            errs = [float(sub.loc[n, f"{metric}_std"]) if n in sub.index and f"{metric}_std" in sub else 0.0 for n in ns]
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                yerr=errs if max(errs) > 0 else None,
                label=policy,
                color=colors[policy],
                edgecolor="black",
                linewidth=1.1,
                hatch=HATCHES.get(policy, ""),
                error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
                zorder=3,
            )
            if metric == "goodput_tok_s":
                for bar, value in zip(bars, vals):
                    if value == 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, 0.9, "0", ha="center", va="bottom", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("Concurrent requests N")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        style_axis(ax)
    axes[0].set_title("(a) Goodput", loc="left", fontweight="bold")
    axes[1].set_title("(b) Tail latency", loc="left", fontweight="bold")
    axes[2].set_title("(c) SLO attainment", loc="left", fontweight="bold")
    axes[3].set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(handles, labels, loc="center left", fontsize=15, handlelength=1.8, labelspacing=0.9)
    fig.tight_layout(pad=1.0, w_pad=1.45)
    save(fig, outdir, "edge_main_bottleneck_scientific")

    seed7 = read_csv(ROOT / "results/tpds_multiseed_edge_practical/seed_7/processed/scheduler_main_summary.csv")
    seed7 = seed7[seed7["N"].eq(8) & seed7["policy"].isin(policies)].copy()
    comps = [
        ("service_target_frac", "Target", PALETTE["neutral_dark"]),
        ("service_draft_frac", "Draft/RPC", PALETTE["blue_main"]),
        ("service_other_frac", "Other", PALETTE["red_2"]),
    ]
    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    left = np.zeros(len(policies))
    y = np.arange(len(policies))
    seed7 = seed7.set_index("policy")
    for col, label, color in comps:
        vals = np.array([float(seed7.loc[p, col]) if p in seed7.index and col in seed7 else 0.0 for p in policies])
        ax.barh(y, vals, left=left, color=color, edgecolor="black", linewidth=1.1, label=label, height=0.62)
        for idx, value in enumerate(vals):
            if value >= 0.13:
                ax.text(
                    left[idx] + value / 2,
                    idx,
                    f"{value * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if label in {"Target", "Draft/RPC"} else "#222222",
                    fontweight="bold",
                )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(policies)
    ax.set_ylabel("Service-time fraction")
    ax.set_xlabel("Fraction of end-to-end service time")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    style_axis(ax, grid_x=True)
    ax.grid(False, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=12, handlelength=1.7, columnspacing=1.2)
    fig.tight_layout(pad=1.0)
    save(fig, outdir, "edge_service_breakdown_n8_scientific")


def copy_to_main(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.suffix.lower() in {".pdf", ".png", ".svg"}:
            shutil.copy2(path, dst_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="paper_assets/figures_scientific")
    parser.add_argument("--main-outdir", default="main/figures_scientific")
    args = parser.parse_args()
    apply_style()
    outdir = (ROOT / args.outdir).resolve()
    plot_main_scheduler(outdir)
    plot_slo_sensitivity(outdir)
    plot_kv_sweep(outdir)
    plot_route_kv_stress(outdir)
    plot_bestfixed_stress(outdir)
    plot_transfer_gpu_measured(outdir)
    plot_ablation(outdir)
    plot_edge(outdir)
    copy_to_main(outdir, (ROOT / args.main_outdir).resolve())
    print(f"saved scientific-style figures to {outdir}")
    print(f"copied figures to {(ROOT / args.main_outdir).resolve()}")


if __name__ == "__main__":
    main()

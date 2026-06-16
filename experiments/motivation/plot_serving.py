"""IEEE-style split plots for the real serving benchmark."""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["single", "multi_edge_k3", "multi_local_k3"]
DISPLAY = {
    "single": "Cloud only",
    "multi_edge_k3": "Edge+cloud",
    "multi_local_k3": "Cloud draft",
}
COLORS = {
    "single": "#2b2b2b",
    "multi_edge_k3": "#0b4f6c",
    "multi_local_k3": "#8b1e3f",
}
HATCHES = {
    "single": "////",
    "multi_edge_k3": "",
    "multi_local_k3": "\\\\\\\\",
}
MARKERS = {
    "single": "o",
    "multi_edge_k3": "s",
    "multi_local_k3": "^",
}

MAIN_FIGURE_PANELS = [
    ("(a) End-to-end throughput", "throughput_tok_s", "Throughput (tok/s)", "bar", False, None, False),
    ("(b) P99 TPOT under contention", "p99_tpot_ms", "P99 TPOT (ms/token)", "bar", True, None, False),
    ("(c) Server queueing pressure", "p95_queue_s", "P95 queueing delay (s)", "line", True, None, False),
    ("(d) Verification yield", "tokens_per_target_invocation", "Tokens / target call", "bar", False, None, True),
]

SINGLECOL_FIGURE_PANELS = [
    ("(a) Throughput", "throughput_tok_s", "Throughput (tok/s)", "bar", False, None, False),
    ("(b) P99 TPOT", "p99_tpot_ms", "P99 TPOT (ms/token)", "bar", True, None, False),
    ("(c) Queueing", "p95_queue_s", "P95 queueing delay (s)", "line", True, None, False),
    ("(d) Verify yield", "tokens_per_target_invocation", "Tokens / target call", "bar", False, None, True),
]


def load_rows(path):
    rows = list(csv.DictReader(open(path)))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value) if any(c in value for c in ".eE") else int(value)
            except (TypeError, ValueError):
                pass
    return rows


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#1f1f1f",
        "xtick.color": "#1f1f1f",
        "ytick.color": "#1f1f1f",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.pad_inches": 0.02,
    })


def modes(rows):
    present = {row["mode"] for row in rows}
    return [m for m in ORDER if m in present]


def n_values(rows):
    return sorted({int(row["N"]) for row in rows})


def values(rows, metric, mode, ns):
    by_n = {int(row["N"]): float(row[metric]) for row in rows if row["mode"] == mode}
    return [by_n.get(n, np.nan) for n in ns]


def with_speedup(rows):
    single = {int(row["N"]): float(row["throughput_tok_s"]) for row in rows if row["mode"] == "single"}
    out = []
    for row in rows:
        copied = dict(row)
        base = single.get(int(row["N"]), np.nan)
        copied["speedup_vs_single"] = float(row["throughput_tok_s"]) / base if base > 0 else np.nan
        tokens = float(row.get("tokens", 0.0))
        target_forwards = float(row.get("target_forwards", 0.0))
        copied["target_invocations_per_token"] = target_forwards / tokens if tokens > 0 else np.nan
        copied["tokens_per_target_invocation"] = tokens / target_forwards if target_forwards > 0 else np.nan
        out.append(copied)
    return out


def _style_axes(ax):
    ax.grid(axis="y", linestyle=(0, (1.2, 2.0)), linewidth=0.55, color="#9a9a9a", alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1f1f1f")
        spine.set_linewidth(0.9)


def bars(ax, rows, metric, ylabel, logy=False, annotate=False):
    ns = n_values(rows)
    ms = modes(rows)
    x = np.arange(len(ns))
    width = min(0.68 / max(1, len(ms)), 0.31)
    offset0 = -width * (len(ms) - 1) / 2
    for i, mode in enumerate(ms):
        ys = values(rows, metric, mode, ns)
        ax.bar(
            x + offset0 + i * width,
            ys,
            width,
            color=COLORS.get(mode, "#777777"),
            edgecolor="#111111",
            linewidth=0.65,
            hatch=HATCHES.get(mode, ""),
            label=DISPLAY.get(mode, mode),
            alpha=0.96,
        )
        if annotate:
            for xi, yi in zip(x + offset0 + i * width, ys):
                if np.isfinite(yi):
                    ax.text(xi, yi * 1.035, f"{yi:.2g}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Concurrency $N$")
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    if logy:
        ax.set_yscale("log")
    else:
        ymax = np.nanmax([v for mode in ms for v in values(rows, metric, mode, ns)])
        if ymax > 0:
            ax.set_ylim(0, ymax * (1.32 if annotate else 1.18))


def lines(ax, rows, metric, ylabel, logy=False, ylim=None):
    ns = n_values(rows)
    for mode in modes(rows):
        ax.plot(
            ns,
            values(rows, metric, mode, ns),
            color=COLORS.get(mode, "#777777"),
            marker=MARKERS.get(mode, "o"),
            linewidth=1.8,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.05,
            label=DISPLAY.get(mode, mode),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Concurrency $N$")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=(0, (1.2, 2.0)), linewidth=0.55, color="#9a9a9a", alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1f1f1f")
        spine.set_linewidth(0.9)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)


def save(fig, outdir, name):
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def default_input(root):
    candidates = [
        root / "results_archive" / "20260603_172101_before_rerun_n64_check" / "serving_benchmark.csv",
        root / "experiments" / "real_matrix" / "results" / "serving_realedge_Qwen2.5-0.5B-Instruct_to_Qwen2.5-7B-Instruct.csv",
        root / "experiments" / "motivation" / "results" / "serving_benchmark.csv",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[-1])


def add_legend(ax):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=max(1, len(modes_cache(ax))),
        handlelength=1.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def modes_cache(ax):
    handles, _ = ax.get_legend_handles_labels()
    return handles


def combined_legend(fig, axes):
    by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label.update(dict(zip(labels, handles)))
    if not by_label:
        return
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(by_label),
        frameon=False,
        handlelength=1.45,
        columnspacing=1.25,
    )


def plot_combined_figure(rows, outdir, name, figsize, spacing, panels=MAIN_FIGURE_PANELS):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    flat_axes = axes.ravel()
    for ax, (title, metric, ylabel, kind, logy, ylim, annotate) in zip(flat_axes, panels):
        if kind == "bar":
            bars(ax, rows, metric, ylabel, logy=logy, annotate=annotate)
            if ylim:
                ax.set_ylim(*ylim)
        else:
            lines(ax, rows, metric, ylabel, logy=logy, ylim=ylim)
        ax.set_title(title, loc="left", pad=2.0, fontsize=8.5, fontweight="bold")
    combined_legend(fig, flat_axes)
    fig.subplots_adjust(**spacing)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    experiment_dir = os.path.abspath(os.path.dirname(__file__))
    root = Path(experiment_dir).resolve().parents[1]
    parser.add_argument("--input", default=default_input(root))
    parser.add_argument("--outdir", default=os.path.join(experiment_dir, "figures"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup()
    rows = with_speedup(load_rows(args.input))

    panels = [
        ("figure2a_serving_throughput", "throughput_tok_s", "Throughput (tok/s)", "bar", False, None, False),
        ("figure2b_serving_speedup", "speedup_vs_single", "Throughput / cloud-only", "line", False, None, False),
        ("figure2c_serving_latency", "mean_latency_s", "Mean latency (s)", "line", True, None, False),
        ("figure2d_serving_p99_tpot", "p99_tpot_ms", "P99 TPOT (ms/token)", "bar", True, None, False),
        ("figure2e_serving_queue", "p95_queue_s", "P95 queueing delay (s)", "line", True, None, False),
        ("figure2f_serving_acceptance", "acceptance_rate", "Draft acceptance rate", "bar", False, (0, 1.02), False),
        ("figure2g_serving_target_calls", "target_invocations_per_token", "Target calls / token", "bar", False, None, True),
        ("figure2h_serving_verify_yield", "tokens_per_target_invocation", "Tokens / target call", "bar", False, None, True),
    ]
    for name, metric, ylabel, kind, logy, ylim, annotate in panels:
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.15))
        if kind == "bar":
            bars(ax, rows, metric, ylabel, logy=logy, annotate=annotate)
            if ylim:
                ax.set_ylim(*ylim)
        else:
            lines(ax, rows, metric, ylabel, logy=logy, ylim=ylim)
        add_legend(ax)
        save(fig, args.outdir, name)

    plot_combined_figure(
        rows,
        args.outdir,
        "figure1_motivation",
        (7.05, 3.75),
        {"left": 0.078, "right": 0.995, "bottom": 0.105, "top": 0.895, "hspace": 0.58, "wspace": 0.34},
    )
    plot_combined_figure(
        rows,
        args.outdir,
        "figure1_motivation_singlecol",
        (3.55, 4.9),
        {"left": 0.185, "right": 0.995, "bottom": 0.075, "top": 0.925, "hspace": 0.70, "wspace": 0.70},
        SINGLECOL_FIGURE_PANELS,
    )

    print(f"loaded serving benchmark from {args.input}")
    print(f"saved serving benchmark split figures to {args.outdir}")


if __name__ == "__main__":
    main()

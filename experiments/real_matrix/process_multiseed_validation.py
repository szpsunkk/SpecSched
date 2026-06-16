"""Aggregate focused multi-seed validation results into CSV and LaTeX tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "goodput_tok_s",
    "p99_tpot_ms",
    "slo_attainment",
    "reprefills_per_1k_tokens",
    "kv_evictions_per_1k_tokens",
    "target_forwards_per_token",
    "x_peak_gpu_mb",
)

SCENARIO_LABELS = {
    "paper3090_transfer_shift": "Workload shift",
    "paper3090_transfer_bursty": "Bursty arrivals",
    "paper3090_transfer_tiered": "Tiered SLOs",
}


def load_config(path: Path) -> dict:
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    cfg = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    if cfg.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema_version={cfg.get('schema_version')!r}")
    return cfg


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return float("nan"), float("nan")
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def ci95(mean: float, std: float, count: int) -> Tuple[float, float, float]:
    if count <= 1 or math.isnan(std):
        return float("nan"), float("nan"), float("nan")
    t975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(count - 1, 1.96)
    half = t975 * std / math.sqrt(count)
    return mean - half, mean + half, half


def summary_files(cfg: dict) -> List[str]:
    names = cfg.get("summary_files", ["scheduler_main_summary.csv"])
    if isinstance(names, str):
        return [names]
    return [str(name) for name in names]


def selected_tags(cfg: dict, experiments_by_tag: Dict[str, dict]) -> set[str]:
    include = {str(value) for value in cfg.get("include_experiments", [])}
    if not include:
        return set(experiments_by_tag)
    selected = set()
    for tag, exp in experiments_by_tag.items():
        keys = {str(exp.get("name", "")), str(exp.get("tag", "")), str(exp.get("kind", ""))}
        if include & keys:
            selected.add(tag)
    return selected


def row_selected(row: dict, allowed_tags: set[str]) -> bool:
    if not allowed_tags:
        return True
    return str(row.get("tag", "")) in allowed_tags


def collect_rows(cfg: dict, raw_fallback: bool = False) -> List[dict]:
    rows: List[dict] = []
    experiments_by_tag = {}
    base_config_path = cfg.get("base_config")
    if base_config_path:
        base = load_config(Path(base_config_path).resolve())
        experiments_by_tag = {str(exp.get("tag", "")): exp for exp in base.get("experiments", [])}
    allowed_tags = selected_tags(cfg, experiments_by_tag)
    for item in cfg.get("reference_results", []):
        seed = int(item["seed"])
        root = Path(item["processed_summary"])
        for row in read_csv(root):
            row = normalize_row(row)
            if not row_selected(row, allowed_tags):
                continue
            row["seed"] = seed
            row["source_summary"] = str(root)
            rows.append(row)
    for seed in cfg["seeds"]:
        processed_root = Path(cfg["results_root"]) / f"seed_{int(seed)}" / "processed"
        found_processed = False
        for summary_name in summary_files(cfg):
            root = processed_root / summary_name
            for row in read_csv(root):
                found_processed = True
                row = normalize_row(row)
                if not row_selected(row, allowed_tags):
                    continue
                row["seed"] = int(seed)
                row["source_summary"] = str(root)
                rows.append(row)
        if raw_fallback and not found_processed:
            raw_root = Path(cfg["results_root"]) / f"seed_{int(seed)}" / "raw"
            for path in sorted(raw_root.glob("*/scheduler_*.csv")):
                if path.name.startswith("scheduler_requests_") or path.name.startswith("scheduler_calib_"):
                    continue
                tag = path.parent.name
                exp = experiments_by_tag.get(tag, {})
                for row in read_csv(path):
                    row = normalize_row(row)
                    row.setdefault("tag", tag)
                    row.setdefault("kind", exp.get("kind", ""))
                    row.setdefault("configured_beta", exp.get("beta", row.get("x_beta", "")))
                    row.setdefault("configured_kv_blocks", exp.get("kv_blocks", row.get("x_kv_blocks", "")))
                    if not row_selected(row, allowed_tags):
                        continue
                    row["seed"] = int(seed)
                    row["source_summary"] = str(path)
                    rows.append(row)
    return rows


def expected_seeds(cfg: dict) -> List[int]:
    refs = [int(item["seed"]) for item in cfg.get("reference_results", [])]
    return refs + [int(seed) for seed in cfg["seeds"]]


def group_fields(cfg: dict) -> List[str]:
    return [str(field) for field in cfg.get("group_fields", [])]


def normalize_row(row: dict) -> dict:
    out = dict(row)
    if not out.get("configured_kv_blocks") and out.get("x_kv_blocks"):
        out["configured_kv_blocks"] = out["x_kv_blocks"]
    if not out.get("configured_beta") and out.get("x_beta"):
        out["configured_beta"] = out["x_beta"]
    return out


def group_tuple_from_row(row: dict, fields: List[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def group_label(fields: List[str], values: Tuple[str, ...]) -> str:
    if not fields:
        return ""
    return ",".join(f"{field}={value}" for field, value in zip(fields, values))


def expected_group_tuples(cfg: dict, rows: List[dict]) -> List[Tuple[str, ...]]:
    fields = group_fields(cfg)
    configured = cfg.get("expected_groups", [])
    if configured:
        return [tuple(str(item.get(field, "")) for field in fields) for item in configured]
    if not fields:
        return [tuple()]
    seen = sorted({group_tuple_from_row(row, fields) for row in rows})
    return seen or [tuple("" for _ in fields)]


def completion_gaps(
    rows: List[dict],
    policies: List[str],
    ns_values: List[int],
    seeds: List[int],
    cfg: dict,
) -> List[str]:
    fields = group_fields(cfg)
    groups = expected_group_tuples(cfg, rows)
    present: Dict[Tuple[Tuple[str, ...], int, str], set] = defaultdict(set)
    for row in rows:
        try:
            n_value = int(float(row.get("N", "")))
            seed = int(row["seed"])
        except (KeyError, ValueError):
            continue
        policy = str(row.get("policy", ""))
        if n_value in ns_values and policy in policies:
            present[(group_tuple_from_row(row, fields), n_value, policy)].add(seed)

    missing = []
    expected = set(seeds)
    for group in groups:
        prefix = f"{group_label(fields, group)}:" if fields else ""
        for n_value in ns_values:
            for policy in policies:
                have = present.get((group, n_value, policy), set())
                if have != expected:
                    missing_seeds = ",".join(str(seed) for seed in seeds if seed not in have)
                    missing.append(f"{prefix}N={n_value}:{policy}:seed={missing_seeds}")
    return missing


def validate_complete(rows: List[dict], policies: List[str], ns_values: List[int], seeds: List[int], cfg: dict) -> None:
    missing = completion_gaps(rows, policies, ns_values, seeds, cfg)
    if missing:
        detail = "; ".join(missing[:24])
        if len(missing) > 24:
            detail += f"; ... +{len(missing) - 24} more"
        raise SystemExit(f"incomplete multi-seed results: {detail}")


def aggregate(rows: List[dict], policies: List[str], ns_values: List[int], cfg: dict) -> List[dict]:
    fields = group_fields(cfg)
    groups = expected_group_tuples(cfg, rows)
    grouped: Dict[Tuple[Tuple[str, ...], int, str], List[dict]] = defaultdict(list)
    for row in rows:
        try:
            n_value = int(float(row.get("N", "")))
        except ValueError:
            continue
        policy = str(row.get("policy", ""))
        if n_value in ns_values and policy in policies:
            grouped[(group_tuple_from_row(row, fields), n_value, policy)].append(row)

    out: List[dict] = []
    for group_values in groups:
        for n_value in ns_values:
            for policy in policies:
                group = grouped.get((group_values, n_value, policy), [])
                if not group:
                    continue
                item = {field: value for field, value in zip(fields, group_values)}
                item.update({
                    "N": n_value,
                    "policy": policy,
                    "seeds": ",".join(str(r["seed"]) for r in group),
                    "count": len(group),
                })
                arrivals = sorted({str(r.get("arrival_model", "")) for r in group if str(r.get("arrival_model", ""))})
                rates = sorted({str(r.get("arrival_rate_rps", "")) for r in group if str(r.get("arrival_rate_rps", ""))})
                item["arrival_model"] = ",".join(arrivals)
                item["arrival_rate_rps"] = ",".join(rates)
                for metric in METRICS:
                    vals = [float(r[metric]) for r in group if r.get(metric) not in ("", None)]
                    if not vals:
                        continue
                    mean, std = mean_std(vals)
                    item[f"{metric}_mean"] = f"{mean:.4f}"
                    item[f"{metric}_std"] = f"{std:.4f}"
                out.append(item)
    return out


def lift_rows(summary_rows: List[dict], cfg: dict, focus_policy: str = "SpecSched") -> List[dict]:
    fields = group_fields(cfg)
    by_cell = {(group_tuple_from_row(row, fields), int(row["N"]), str(row["policy"])): row for row in summary_rows}
    out: List[dict] = []
    fixed = ("FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7")
    configured_baselines = [str(policy) for policy in cfg.get("paired_baselines", [])]
    groups = sorted({group_tuple_from_row(row, fields) for row in summary_rows}) or [tuple()]
    for group_values in groups:
        for n_value in sorted({int(row["N"]) for row in summary_rows}):
            focus = by_cell.get((group_values, n_value, focus_policy))
            if not focus:
                continue
            focus_goodput = float(focus["goodput_tok_s_mean"])
            best_fixed = None
            for policy in fixed:
                row = by_cell.get((group_values, n_value, policy))
                if row and (best_fixed is None or float(row["goodput_tok_s_mean"]) > float(best_fixed["goodput_tok_s_mean"])):
                    best_fixed = row
            baselines = [
                ("NoSpec", by_cell.get((group_values, n_value, "NoSpec"))),
                ("TETRIS-Proxy", by_cell.get((group_values, n_value, "TETRIS-Proxy"))),
            ]
            for baseline_policy in configured_baselines:
                baselines.append((baseline_policy, by_cell.get((group_values, n_value, baseline_policy))))
            if best_fixed:
                baselines.append((f"BestFixed:{best_fixed['policy']}", best_fixed))
            for label, base in baselines:
                if not base:
                    continue
                base_goodput = float(base["goodput_tok_s_mean"])
                lift = (focus_goodput / base_goodput - 1.0) * 100.0 if base_goodput > 0 else float("nan")
                item = {field: value for field, value in zip(fields, group_values)}
                item.update({
                "N": n_value,
                "focus_policy": focus_policy,
                "baseline": label,
                "focus_goodput_mean": f"{focus_goodput:.4f}",
                "baseline_goodput_mean": f"{base_goodput:.4f}",
                "goodput_lift_pct": f"{lift:.4f}",
                })
                out.append(item)
    return out


def paired_lift_rows(rows: List[dict], policies: List[str], ns_values: List[int], cfg: dict, focus_policy: str = "SpecSched") -> List[dict]:
    fields = group_fields(cfg)
    by_cell = {}
    fixed = ("FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7")
    for row in rows:
        try:
            seed = int(row["seed"])
            n_value = int(float(row.get("N", "")))
        except (KeyError, ValueError):
            continue
        policy = str(row.get("policy", ""))
        if n_value in ns_values and policy in policies:
            by_cell[(seed, group_tuple_from_row(row, fields), n_value, policy)] = row

    out: List[dict] = []
    seeds = sorted({seed for seed, _, _, _ in by_cell})
    groups = expected_group_tuples(cfg, rows)
    configured_baselines = [str(policy) for policy in cfg.get("paired_baselines", [])]
    baseline_order = ["NoSpec", "TETRIS-Proxy", "BestFixed"] + configured_baselines
    for group_values in groups:
        for n_value in ns_values:
            for baseline in baseline_order:
                lifts = []
                baseline_labels = []
                for seed in seeds:
                    focus = by_cell.get((seed, group_values, n_value, focus_policy))
                    if not focus:
                        continue
                    if baseline == "BestFixed":
                        base = None
                        for policy in fixed:
                            row = by_cell.get((seed, group_values, n_value, policy))
                            if row and (
                                base is None
                                or float(row["goodput_tok_s"]) > float(base["goodput_tok_s"])
                            ):
                                base = row
                        label = f"BestFixed:{base['policy']}" if base else ""
                    else:
                        base = by_cell.get((seed, group_values, n_value, baseline))
                        label = baseline
                    if not base:
                        continue
                    base_goodput = float(base["goodput_tok_s"])
                    if base_goodput <= 0:
                        continue
                    lifts.append((float(focus["goodput_tok_s"]) / base_goodput - 1.0) * 100.0)
                    baseline_labels.append(label)
                if not lifts:
                    continue
                mean, std = mean_std(lifts)
                lo, hi, half = ci95(mean, std, len(lifts))
                item = {field: value for field, value in zip(fields, group_values)}
                item.update({
                "N": n_value,
                "focus_policy": focus_policy,
                "baseline": baseline,
                "seeds": ",".join(str(seed) for seed in seeds if (seed, group_values, n_value, focus_policy) in by_cell),
                "baseline_labels": ",".join(baseline_labels),
                "count": len(lifts),
                "goodput_lift_pct_mean": f"{mean:.4f}",
                "goodput_lift_pct_std": f"{std:.4f}",
                "goodput_lift_pct_ci95_low": f"{lo:.4f}",
                "goodput_lift_pct_ci95_high": f"{hi:.4f}",
                "goodput_lift_pct_ci95_half_width": f"{half:.4f}",
                "ci95_positive": int(lo > 0),
                })
                out.append(item)
    return out


def per_seed_bestfixed_rows(rows: List[dict], fields: List[str], group_values: Tuple[str, ...], n_value: int) -> List[dict]:
    fixed = ("FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7")
    by_seed: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        try:
            seed = int(row["seed"])
            row_n = int(float(row.get("N", "")))
        except (KeyError, ValueError):
            continue
        if row_n != n_value or group_tuple_from_row(row, fields) != group_values:
            continue
        if str(row.get("policy", "")) in fixed:
            by_seed[seed].append(row)
    best_rows = []
    for seed in sorted(by_seed):
        best_rows.append(max(by_seed[seed], key=lambda row: float(row["goodput_tok_s"])))
    return best_rows


def write_transfer_summary(
    summary_rows: List[dict],
    paired_rows: List[dict],
    raw_rows: List[dict],
    cfg: dict,
    processed: Path,
) -> None:
    fields = group_fields(cfg)
    if "tag" not in fields:
        return
    tag_idx = fields.index("tag")
    fixed = ("FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7")
    by_cell = {
        (group_tuple_from_row(row, fields), int(row["N"]), str(row["policy"])): row
        for row in summary_rows
    }
    paired_by_cell = {
        (group_tuple_from_row(row, fields), int(row["N"]), str(row["baseline"])): row
        for row in paired_rows
    }
    out: List[dict] = []
    for group_values in sorted({group_tuple_from_row(row, fields) for row in summary_rows}):
        tag = group_values[tag_idx]
        if tag not in SCENARIO_LABELS:
            continue
        for n_value in sorted({int(row["N"]) for row in summary_rows}):
            spec = by_cell.get((group_values, n_value, "SpecSched"))
            if not spec:
                continue
            best = None
            for policy in fixed:
                row = by_cell.get((group_values, n_value, policy))
                if row and (
                    best is None
                    or float(row["goodput_tok_s_mean"]) > float(best["goodput_tok_s_mean"])
                ):
                    best = row
            paired = paired_by_cell.get((group_values, n_value, "BestFixed"))
            if not best or not paired:
                continue
            per_seed_best = per_seed_bestfixed_rows(raw_rows, fields, group_values, n_value)
            best_goodputs = [float(row["goodput_tok_s"]) for row in per_seed_best if row.get("goodput_tok_s")]
            best_p99s = [float(row["p99_tpot_ms"]) for row in per_seed_best if row.get("p99_tpot_ms")]
            best_mean, best_std = mean_std(best_goodputs)
            best_p99_mean, _ = mean_std(best_p99s)
            best_policies = sorted({str(row.get("policy", "")) for row in per_seed_best})
            if not best_goodputs:
                best_mean = float(best.get("goodput_tok_s_mean", "nan"))
                best_std = float(best.get("goodput_tok_s_std", "nan"))
            if not best_p99s:
                best_p99_mean = float(best.get("p99_tpot_ms_mean", "nan"))
            noslo = by_cell.get((group_values, n_value, "SpecSched-NoSLO"))
            row = {
                "scenario": SCENARIO_LABELS[tag],
                "tag": tag,
                "N": n_value,
                "bestfixed_policy": best_policies[0] if len(best_policies) == 1 else "per-seed",
                "bestfixed_policy_by_seed": ",".join(
                    f"{int(seed_row['seed'])}:{seed_row['policy']}" for seed_row in per_seed_best
                ),
                "bestfixed_goodput_tok_s_mean": f"{best_mean:.4f}",
                "bestfixed_goodput_tok_s_std": f"{best_std:.4f}",
                "specsched_goodput_tok_s_mean": spec.get("goodput_tok_s_mean", ""),
                "specsched_goodput_tok_s_std": spec.get("goodput_tok_s_std", ""),
                "specsched_lift_vs_bestfixed_pct_mean": paired.get("goodput_lift_pct_mean", ""),
                "specsched_lift_vs_bestfixed_pct_ci95_low": paired.get("goodput_lift_pct_ci95_low", ""),
                "specsched_lift_vs_bestfixed_pct_ci95_high": paired.get("goodput_lift_pct_ci95_high", ""),
                "specsched_lift_vs_bestfixed_pct_ci95_half_width": paired.get("goodput_lift_pct_ci95_half_width", ""),
                "bestfixed_p99_tpot_ms_mean": f"{best_p99_mean:.4f}",
                "specsched_p99_tpot_ms_mean": spec.get("p99_tpot_ms_mean", ""),
                "specsched_slo_attainment_mean": spec.get("slo_attainment_mean", ""),
                "seeds": spec.get("seeds", ""),
                "source": "measured_gpu_multiseed",
            }
            if noslo:
                noslo_goodput = float(noslo.get("goodput_tok_s_mean", "nan"))
                spec_goodput = float(spec.get("goodput_tok_s_mean", "nan"))
                row["noslo_goodput_tok_s_mean"] = noslo.get("goodput_tok_s_mean", "")
                row["specsched_lift_vs_noslo_pct_mean"] = (
                    f"{((spec_goodput / noslo_goodput - 1.0) * 100.0):.4f}"
                    if noslo_goodput > 0
                    else ""
                )
            out.append(row)
    if not out:
        return
    write_csv(processed / "transfer_gpu_multiseed_bestfixed_summary.csv", out)
    write_transfer_latex(ROOT / "paper_assets" / "tables" / "transfer_gpu_measured_multiseed.tex", out)


def write_transfer_latex(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Measured three-seed non-stationary transfer replay at $N=64$. BestFixed is the best full-run fixed depth from $k\\in\\{1,3,5,7\\}$ selected per seed; intervals are paired 95\\% CIs for \\system{} lift over BestFixed.}",
        "\\label{tab:transfer-gpu-measured}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Scenario & BestFixed & \\system{} & Lift & \\system{} P99 \\\\",
        "\\midrule",
    ]
    for row in rows:
        lift = float(row["specsched_lift_vs_bestfixed_pct_mean"])
        lo = float(row["specsched_lift_vs_bestfixed_pct_ci95_low"])
        hi = float(row["specsched_lift_vs_bestfixed_pct_ci95_high"])
        lines.append(
            f"{row['scenario']} & "
            f"{float(row['bestfixed_goodput_tok_s_mean']):.1f} & "
            f"{float(row['specsched_goodput_tok_s_mean']):.1f} & "
            f"{lift:+.1f}\\% [{lo:+.1f},{hi:+.1f}] & "
            f"{float(row['specsched_p99_tpot_ms_mean']):.0f} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{-0.5em}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def partial_diagnostics(rows: List[dict], policies: List[str], ns_values: List[int], seeds: List[int], cfg: dict) -> List[dict]:
    fields = group_fields(cfg)
    groups = expected_group_tuples(cfg, rows)
    by_seed_group_n: Dict[Tuple[int, Tuple[str, ...], int], List[dict]] = defaultdict(list)
    fixed = ("FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7")
    wanted = set(policies)
    for row in rows:
        try:
            seed = int(row["seed"])
            n_value = int(float(row.get("N", "")))
        except (KeyError, ValueError):
            continue
        if n_value in ns_values and str(row.get("policy", "")) in wanted:
            by_seed_group_n[(seed, group_tuple_from_row(row, fields), n_value)].append(row)

    out: List[dict] = []
    for group_values in groups:
        group_meta = {field: value for field, value in zip(fields, group_values)}
        for seed in seeds:
            for n_value in ns_values:
                group = by_seed_group_n.get((seed, group_values, n_value), [])
                by_policy = {str(row.get("policy", "")): row for row in group}
                completed = sorted(by_policy)
                best_fixed = None
                for policy in fixed:
                    row = by_policy.get(policy)
                    if row and (
                        best_fixed is None
                        or float(row["goodput_tok_s"]) > float(best_fixed["goodput_tok_s"])
                    ):
                        best_fixed = row
                spec = by_policy.get("SpecSched")
                item = dict(group_meta)
                item.update({
                    "seed": seed,
                    "N": n_value,
                    "completed_count": len(completed),
                    "expected_count": len(policies),
                    "completed_policies": ",".join(completed),
                    "has_specsched": int(spec is not None),
                    "best_fixed_policy": str(best_fixed.get("policy", "")) if best_fixed else "",
                    "best_fixed_goodput_tok_s": f"{float(best_fixed['goodput_tok_s']):.4f}" if best_fixed else "",
                    "specsched_goodput_tok_s": f"{float(spec['goodput_tok_s']):.4f}" if spec else "",
                    "specsched_vs_bestfixed_pct": "",
                    "diagnostic_note": "partial diagnostic only; do not cite",
                })
                if spec and best_fixed:
                    lift = (float(spec["goodput_tok_s"]) / float(best_fixed["goodput_tok_s"]) - 1.0) * 100.0
                    item["specsched_vs_bestfixed_pct"] = f"{lift:.4f}"
                out.append(item)
    return out


def latex_table(rows: List[dict], path: Path) -> None:
    keep = [row for row in rows if row["policy"] in {"NoSpec", "TETRIS-Proxy", "SpecSched", "SpecSched-PressureAIMD"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "$N$ & Policy & Goodput & P99 TPOT & SLO att. \\\\",
        "\\midrule",
    ]
    for row in keep:
        lines.append(
            f"{row['N']} & {row['policy']} & "
            f"{float(row['goodput_tok_s_mean']):.2f}$\\pm${float(row['goodput_tok_s_std']):.2f} & "
            f"{float(row['p99_tpot_ms_mean']):.1f}$\\pm${float(row['p99_tpot_ms_std']):.1f} & "
            f"{float(row['slo_attainment_mean']):.3f}$\\pm${float(row['slo_attainment_std']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tpds_multiseed_main.json")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write diagnostic partial aggregates from available processed/raw rows; not for paper claims",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    rows = collect_rows(cfg, raw_fallback=args.allow_partial)
    if not rows:
        raise SystemExit("no processed multi-seed rows found; run run_multiseed_validation.py first")

    policies = [str(policy) for policy in cfg.get("aggregate_policies", cfg["policies"])]
    ns_values = [int(n) for n in cfg["Ns"]]
    seeds = expected_seeds(cfg)
    focus_policy = str(cfg.get("focus_policy", "SpecSched"))
    missing = completion_gaps(rows, policies, ns_values, seeds, cfg)
    if missing and not args.allow_partial:
        validate_complete(rows, policies, ns_values, seeds, cfg)
    if missing and args.allow_partial:
        detail = "; ".join(missing[:12])
        if len(missing) > 12:
            detail += f"; ... +{len(missing) - 12} more"
        print(f"[multiseed][partial] incomplete results: {detail}", file=sys.stderr)
    out_rows = aggregate(rows, policies, ns_values, cfg)
    if not out_rows:
        raise SystemExit("no rows matched the configured multi-seed policy/N filters")

    processed = Path(cfg["results_root"]) / ("processed_partial" if args.allow_partial else "processed")
    write_csv(processed / "main_multiseed_summary.csv", out_rows)
    write_csv(processed / "main_multiseed_lifts.csv", lift_rows(out_rows, cfg, focus_policy))
    paired_rows = paired_lift_rows(rows, policies, ns_values, cfg, focus_policy)
    write_csv(processed / "main_multiseed_paired_lifts.csv", paired_rows)
    if not args.allow_partial:
        write_transfer_summary(out_rows, paired_rows, rows, cfg, processed)
    if args.allow_partial:
        write_csv(processed / "partial_diagnostics.csv", partial_diagnostics(rows, policies, ns_values, seeds, cfg))
    table_dir = Path(cfg["results_root"]) / ("tables_partial" if args.allow_partial else "tables")
    latex_table(out_rows, table_dir / "main_multiseed_table.tex")
    print(f"[multiseed] wrote {processed / 'main_multiseed_summary.csv'}")


if __name__ == "__main__":
    main()

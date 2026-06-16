#!/usr/bin/env python3
"""Calibrated stress diagnostic for BestFixed transfer limits.

The script reads measured multi-seed scheduler summaries and constructs three
stress scenarios from improve.md: workload shift, bursty arrivals, and tiered
SLOs. It is intentionally a result-level diagnostic, not a GPU rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value.replace("${PROJECT_ROOT}", str(ROOT)))).expanduser()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    if int(cfg.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported schema_version in {path}")
    return cfg


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def measured_baseline(cfg: dict[str, Any]) -> dict[tuple[int, str], dict[str, float]]:
    rows = read_csv(expand_path(cfg["paths"]["source_main_summary"]))
    policies = set(cfg["simulation"]["fixed_policies"]) | {"SpecSched"}
    base: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        policy = row.get("policy", "")
        if policy not in policies:
            continue
        n_value = int(float(row["N"]))
        if n_value not in {int(v) for v in cfg["simulation"]["concurrency_N"]}:
            continue
        base[(n_value, policy)] = {
            "goodput": as_float(row, "goodput_tok_s_mean"),
            "p99": as_float(row, "p99_tpot_ms_mean"),
            "slo": as_float(row, "slo_attainment_mean"),
            "evictions": as_float(row, "kv_evictions_per_1k_tokens_mean"),
            "reprefills": as_float(row, "reprefills_per_1k_tokens_mean"),
        }
    missing = [
        (int(n_value), policy)
        for n_value in cfg["simulation"]["concurrency_N"]
        for policy in policies
        if (int(n_value), policy) not in base
    ]
    if missing:
        raise ValueError(f"Missing measured baseline cells: {missing}")
    return base


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_metric(values: list[float], weights: list[float]) -> float:
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def simulate(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sim = cfg["simulation"]
    rng = random.Random(int(cfg["experiment"]["seed"]))
    base = measured_baseline(cfg)
    raw_rows: list[dict[str, Any]] = []

    for n_value in [int(v) for v in sim["concurrency_N"]]:
        global_policy = str(sim["global_fixed_policy"][str(n_value)])
        for scenario_name, scenario in sim["scenarios"].items():
            segments = [str(v) for v in scenario["segments"]]
            weights = [float(v) for v in scenario["segment_weights"]]
            penalties = scenario["fixed_mismatch_penalty"]
            phase_best = scenario["best_fixed_by_segment"]
            spec_rel = float(scenario["specsched_relative_to_phase_best"])
            recovery_s = float(scenario["specsched_recovery_s"])

            for rep in range(int(sim["replicates"])):
                jitter_g = 1.0 + rng.uniform(-float(sim["max_goodput_jitter"]), float(sim["max_goodput_jitter"]))
                jitter_l = 1.0 + rng.uniform(-float(sim["max_latency_jitter"]), float(sim["max_latency_jitter"]))

                for method in sim["methods"]:
                    seg_goodputs: list[float] = []
                    seg_p99s: list[float] = []
                    seg_attain: list[float] = []
                    seg_evictions: list[float] = []
                    selected: list[str] = []

                    for segment in segments:
                        if method == "BestFixed-Global":
                            policy = global_policy
                            factor = float(penalties[segment][policy])
                            adapt_penalty = 1.0
                        elif method == "BestFixed-PhaseTuned":
                            policy = str(phase_best[segment])
                            factor = float(penalties[segment][policy])
                            adapt_penalty = 1.0
                        elif method == "SpecSched":
                            policy = str(phase_best[segment])
                            factor = float(penalties[segment][policy]) * spec_rel
                            adapt_penalty = 0.985 if segment in {"low_acceptance", "burst", "tight"} else 1.0
                        else:
                            raise ValueError(f"unknown method {method}")

                        b = base[(n_value, policy)]
                        selected.append(policy)
                        pressure = 1.0 / max(factor, 0.01)
                        seg_goodputs.append(b["goodput"] * factor * jitter_g)
                        seg_p99s.append(b["p99"] * (1.0 + 0.42 * max(0.0, pressure - 1.0)) * jitter_l / adapt_penalty)
                        seg_attain.append(clamp(b["slo"] - 0.09 * max(0.0, pressure - 1.0), 0.0, 1.0))
                        seg_evictions.append(b["evictions"] * (1.0 + 0.65 * max(0.0, pressure - 1.0)))

                    goodput = weighted_metric(seg_goodputs, weights)
                    p99 = weighted_metric(seg_p99s, weights)
                    slo = weighted_metric(seg_attain, weights)
                    evictions = weighted_metric(seg_evictions, weights)
                    recovery = 0.0 if method.startswith("BestFixed") else recovery_s
                    if method == "BestFixed-Global" and scenario_name in {"workload_shift", "bursty_arrival"}:
                        recovery = recovery_s * 2.5
                    elif method == "BestFixed-PhaseTuned":
                        recovery = recovery_s * 0.4

                    raw_rows.append(
                        {
                            "experiment": cfg["experiment"]["name"],
                            "data_kind": "calibrated_simulation",
                            "seed": int(cfg["experiment"]["seed"]),
                            "replicate": rep,
                            "scenario": scenario_name,
                            "N": n_value,
                            "method": method,
                            "selected_policies": ";".join(selected),
                            "goodput_tok_s": round(goodput, 6),
                            "p99_tpot_ms": round(p99, 6),
                            "slo_attainment": round(slo, 6),
                            "kv_evictions_per_1k_tokens": round(evictions, 6),
                            "recovery_time_s": round(recovery, 6),
                        }
                    )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ["scenario", "N", "method"]
    for row in raw_rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        out = dict(zip(keys, key))
        out["count"] = len(rows)
        out["selected_policies"] = rows[0]["selected_policies"]
        for metric in [
            "goodput_tok_s",
            "p99_tpot_ms",
            "slo_attainment",
            "kv_evictions_per_1k_tokens",
            "recovery_time_s",
        ]:
            vals = [float(row[metric]) for row in rows]
            out[f"{metric}_mean"] = round(mean(vals), 6)
            out[f"{metric}_std"] = round(pstdev(vals), 6)
        summary_rows.append(out)

    manifest = {
        "experiment": cfg["experiment"],
        "source_main_summary": str(expand_path(cfg["paths"]["source_main_summary"])),
        "note": (
            "Calibrated simulation. Local policy baselines are measured; workload-shift, "
            "bursty-arrival, and tiered-SLO transfer penalties are modeled diagnostics."
        ),
    }
    return raw_rows, summary_rows, manifest


def write_table(summary_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if int(row["N"]) == 64 and str(row["method"]) in {"BestFixed-Global", "SpecSched"}
    ]
    order = {"workload_shift": 0, "bursty_arrival": 1, "tiered_slo": 2}
    rows.sort(key=lambda r: (order[str(r["scenario"])], str(r["method"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{BestFixed transfer stress diagnostic at $N=64$. Values are calibrated simulation means over five deterministic replicates.}\n")
        f.write("\\label{tab:bestfixed-stress}\n")
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{3.0pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.12}\n")
        f.write("\\begin{tabular}{@{}llrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("Scenario & Method & Goodput $\\uparrow$ & P99 $\\downarrow$ & Recovery $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        label = {
            "workload_shift": "Shift",
            "bursty_arrival": "Bursty",
            "tiered_slo": "Tiered",
            "BestFixed-Global": "BestFixed",
            "SpecSched": "\\system{}",
        }
        for row in rows:
            f.write(
                f"{label[str(row['scenario'])]} & {label[str(row['method'])]} & "
                f"{float(row['goodput_tok_s_mean']):.1f} & "
                f"{float(row['p99_tpot_ms_mean']):.0f} & "
                f"{float(row['recovery_time_s_mean']):.0f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{-0.5em}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bestfixed_stress_sim.json")
    args = parser.parse_args()
    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    raw_rows, summary_rows, manifest = simulate(cfg)

    raw_dir = expand_path(cfg["paths"]["raw_dir"])
    processed_dir = expand_path(cfg["paths"]["processed_dir"])
    write_csv(raw_dir / "bestfixed_stress_raw.csv", raw_rows)
    write_csv(processed_dir / "bestfixed_stress_summary.csv", summary_rows)
    with (raw_dir / "bestfixed_stress_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    write_table(summary_rows, ROOT / "paper_assets" / "tables" / "bestfixed_stress_sim.tex")
    print(f"wrote {len(raw_rows)} raw rows to {raw_dir / 'bestfixed_stress_raw.csv'}")
    print(f"wrote {len(summary_rows)} summary rows to {processed_dir / 'bestfixed_stress_summary.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calibrated routing x KV x load stress simulation.

This script is intentionally separate from the GPU experiment runner. It reads
measured SpecSched summaries, uses them as the local-route baseline, and applies
deterministic route-delay, remote-hit, migration, and oscillation penalties.
Outputs are labeled as simulation data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def measured_baseline(cfg: dict[str, Any]) -> dict[tuple[int, int], dict[str, float]]:
    sim = cfg["simulation"]
    policy = sim.get("base_policy", "SpecSched")
    main_rows = read_csv(expand_path(cfg["paths"]["source_main_summary"]))
    kv_rows = read_csv(expand_path(cfg["paths"]["source_kv_summary"]))
    base: dict[tuple[int, int], dict[str, float]] = {}

    for row in main_rows:
        if row.get("policy") != policy:
            continue
        n_value = int(float(row["N"]))
        base[(n_value, 1024)] = {
            "goodput": as_float(row, "goodput_tok_s_mean"),
            "p99": as_float(row, "p99_tpot_ms_mean"),
            "slo": as_float(row, "slo_attainment_mean"),
            "evictions": as_float(row, "kv_evictions_per_1k_tokens_mean", 0.0),
            "reprefills": as_float(row, "reprefills_per_1k_tokens_mean", 0.0),
        }

    for row in kv_rows:
        if row.get("policy") != policy:
            continue
        n_value = int(float(row["N"]))
        kv_blocks = int(float(row.get("configured_kv_blocks") or row.get("kv_blocks_total")))
        base[(n_value, kv_blocks)] = {
            "goodput": as_float(row, "goodput_tok_s_mean"),
            "p99": as_float(row, "p99_tpot_ms_mean"),
            "slo": as_float(row, "slo_attainment_mean"),
            "evictions": as_float(row, "kv_evictions_per_1k_tokens_mean", 0.0),
            "reprefills": as_float(row, "reprefills_per_1k_tokens_mean", 0.0),
        }

    # KV-pressure measurements are available at N=64. For lower-N stress
    # cells, anchor goodput/latency to the measured main sweep at that N and
    # apply the N=64 KV relative factor. This keeps the simulation calibrated
    # to observed local-route behavior without inventing an unmeasured GPU cell.
    kv_ref: dict[int, dict[str, float]] = {}
    if (64, 1024) in base:
        ref = base[(64, 1024)]
        for kv in [int(v) for v in sim["kv_blocks"]]:
            if (64, kv) not in base:
                continue
            cur = base[(64, kv)]
            kv_ref[kv] = {
                "goodput": cur["goodput"] / ref["goodput"] if ref["goodput"] else 1.0,
                "p99": cur["p99"] / ref["p99"] if ref["p99"] else 1.0,
                "slo": cur["slo"] / ref["slo"] if ref["slo"] else 1.0,
                "evictions": cur["evictions"],
                "reprefills": cur["reprefills"],
            }
    for n_value in [int(v) for v in sim["concurrency_N"]]:
        if (n_value, 1024) not in base:
            continue
        main = base[(n_value, 1024)]
        for kv in [int(v) for v in sim["kv_blocks"]]:
            if (n_value, kv) in base or kv not in kv_ref:
                continue
            factor = kv_ref[kv]
            base[(n_value, kv)] = {
                "goodput": main["goodput"] * factor["goodput"],
                "p99": main["p99"] * factor["p99"],
                "slo": clamp(main["slo"] * factor["slo"], 0.0, 1.0),
                "evictions": factor["evictions"] * (n_value / 64.0),
                "reprefills": factor["reprefills"] * (n_value / 64.0),
            }

    missing = [
        (n_value, kv)
        for n_value in sim["concurrency_N"]
        for kv in sim["kv_blocks"]
        if (int(n_value), int(kv)) not in base
    ]
    if missing:
        raise ValueError(f"Missing measured or calibrated baseline cells: {missing}")
    return base


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def simulate(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sim = cfg["simulation"]
    rng = random.Random(int(cfg["experiment"]["seed"]))
    base = measured_baseline(cfg)
    raw_rows: list[dict[str, Any]] = []

    for n_value in [int(v) for v in sim["concurrency_N"]]:
        for kv_blocks in [int(v) for v in sim["kv_blocks"]]:
            b = base[(n_value, kv_blocks)]
            kv_pressure = (n_value * 16.0) / float(kv_blocks)
            load_pressure = n_value / 64.0
            for delay_ms in [float(v) for v in sim["route_delays_ms"]]:
                for mode in sim["routing_modes"]:
                    remote_hit_nominal = float(sim["remote_hit_base"][mode])
                    oscillation_nominal = float(sim["oscillation_base"][mode])
                    migration_nominal = float(sim["migration_base"][mode])
                    for rep in range(int(sim["replicates"])):
                        jitter_g = 1.0 + rng.uniform(-float(sim["max_goodput_jitter"]), float(sim["max_goodput_jitter"]))
                        jitter_l = 1.0 + rng.uniform(-float(sim["max_latency_jitter"]), float(sim["max_latency_jitter"]))
                        delay_pressure = delay_ms / 10.0
                        remote_hit = clamp(
                            remote_hit_nominal
                            - 0.06 * delay_pressure
                            - 0.12 * kv_pressure
                            + (0.04 if mode == "oracle" else 0.0),
                            0.0,
                            0.92,
                        )
                        oscillation = clamp(
                            oscillation_nominal
                            + 0.08 * delay_pressure
                            + 0.06 * load_pressure
                            + 0.10 * kv_pressure
                            - (0.04 if mode in {"kv-aware", "oracle"} else 0.0),
                            0.0,
                            0.95,
                        )
                        migration = clamp(
                            migration_nominal
                            + 0.10 * delay_pressure
                            + 0.04 * load_pressure
                            + 0.05 * kv_pressure
                            - 0.07 * remote_hit,
                            0.0,
                            0.95,
                        )

                        remote_benefit = 0.18 * remote_hit * kv_pressure
                        delay_penalty = delay_pressure * (0.10 + 0.18 * migration + 0.14 * oscillation)
                        thrash_penalty = kv_pressure * (0.10 + 0.12 * oscillation)
                        if mode == "local-only":
                            remote_benefit = 0.0
                            delay_penalty = 0.0
                            migration = 0.0
                            oscillation = 0.0

                        goodput = b["goodput"] * clamp(1.0 + remote_benefit - delay_penalty - thrash_penalty, 0.28, 1.18) * jitter_g
                        p99 = b["p99"] * (1.0 + 0.24 * delay_pressure + 0.42 * migration + 0.55 * oscillation + 0.25 * kv_pressure) * jitter_l
                        evictions = b["evictions"] * (1.0 + 0.35 * kv_pressure + 0.65 * oscillation - 0.22 * remote_hit)
                        reprefills = b["reprefills"] * (1.0 + 0.30 * kv_pressure + 0.75 * oscillation - 0.25 * remote_hit)
                        slo = clamp(b["slo"] - 0.00018 * max(0.0, p99 - b["p99"]) - 0.10 * oscillation, 0.0, 1.0)
                        kv_util = clamp(0.55 + 0.55 * kv_pressure + 0.08 * load_pressure - 0.10 * remote_hit, 0.0, 1.0)
                        collapse = int(oscillation > 0.36 and kv_pressure > 0.45 and delay_ms >= 5)

                        raw_rows.append(
                            {
                                "experiment": cfg["experiment"]["name"],
                                "data_kind": "calibrated_simulation",
                                "seed": int(cfg["experiment"]["seed"]),
                                "replicate": rep,
                                "N": n_value,
                                "kv_blocks": kv_blocks,
                                "route_delay_ms": delay_ms,
                                "routing_mode": mode,
                                "base_goodput_tok_s": round(b["goodput"], 6),
                                "base_p99_tpot_ms": round(b["p99"], 6),
                                "goodput_tok_s": round(goodput, 6),
                                "p99_tpot_ms": round(p99, 6),
                                "slo_attainment": round(slo, 6),
                                "kv_utilization": round(kv_util, 6),
                                "remote_hit_ratio": round(remote_hit, 6),
                                "migration_rate": round(migration, 6),
                                "routing_oscillation_rate": round(oscillation, 6),
                                "kv_evictions_per_1k_tokens": round(max(0.0, evictions), 6),
                                "reprefills_per_1k_tokens": round(max(0.0, reprefills), 6),
                                "collapse_flag": collapse,
                            }
                        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ["N", "kv_blocks", "route_delay_ms", "routing_mode"]
    for row in raw_rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        out = dict(zip(keys, key))
        out["count"] = len(rows)
        for metric in [
            "goodput_tok_s",
            "p99_tpot_ms",
            "slo_attainment",
            "kv_utilization",
            "remote_hit_ratio",
            "migration_rate",
            "routing_oscillation_rate",
            "kv_evictions_per_1k_tokens",
            "reprefills_per_1k_tokens",
        ]:
            vals = [float(row[metric]) for row in rows]
            out[f"{metric}_mean"] = round(mean(vals), 6)
            out[f"{metric}_std"] = round(pstdev(vals), 6)
        out["collapse_rate"] = round(mean([float(row["collapse_flag"]) for row in rows]), 6)
        summary_rows.append(out)

    manifest = {
        "experiment": cfg["experiment"],
        "source_main_summary": str(expand_path(cfg["paths"]["source_main_summary"])),
        "source_kv_summary": str(expand_path(cfg["paths"]["source_kv_summary"])),
        "note": "Calibrated simulation. Local baselines are measured; route-delay and remote-KV effects are modeled.",
    }
    return raw_rows, summary_rows, manifest


def write_table(summary_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if int(row["N"]) == 64 and int(row["kv_blocks"]) == 512 and float(row["route_delay_ms"]) == 5.0
    ]
    order = {"local-only": 0, "random": 1, "kv-aware": 2, "oracle": 3}
    rows.sort(key=lambda r: order[str(r["routing_mode"])])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Routing and remote-KV stress simulation at $N=64$, 512 KV blocks, and 5 ms route delay. Values are calibrated simulation means over five deterministic replicates.}\n")
        f.write("\\label{tab:route-kv-sim}\n")
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{3.2pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.12}\n")
        f.write("\\begin{tabular}{@{}lrrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("Route & Goodput $\\uparrow$ & P99 $\\downarrow$ & Hit & Osc. \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['routing_mode']} & "
                f"{float(row['goodput_tok_s_mean']):.1f} & "
                f"{float(row['p99_tpot_ms_mean']):.0f} & "
                f"{float(row['remote_hit_ratio_mean']):.2f} & "
                f"{float(row['routing_oscillation_rate_mean']):.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{-0.5em}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/route_kv_stress_sim.json")
    args = parser.parse_args()
    cfg = load_config((ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    raw_rows, summary_rows, manifest = simulate(cfg)

    raw_dir = expand_path(cfg["paths"]["raw_dir"])
    processed_dir = expand_path(cfg["paths"]["processed_dir"])
    write_csv(raw_dir / "route_kv_stress_raw.csv", raw_rows)
    write_csv(processed_dir / "route_kv_stress_summary.csv", summary_rows)
    with (raw_dir / "route_kv_stress_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    write_table(summary_rows, ROOT / "paper_assets" / "tables" / "route_kv_stress_sim.tex")
    print(f"wrote {len(raw_rows)} raw rows to {raw_dir / 'route_kv_stress_raw.csv'}")
    print(f"wrote {len(summary_rows)} summary rows to {processed_dir / 'route_kv_stress_summary.csv'}")


if __name__ == "__main__":
    main()

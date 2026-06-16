#!/usr/bin/env python3
"""Audit verification-latency model ranking stability under perturbations.

The current scheduler does not read the fitted latency coefficients at runtime,
so this script intentionally reports an offline ranking-stability audit rather
than a goodput-degradation claim. It uses the saved verification-latency fit CSV
and asks whether +/-20% coefficient perturbations preserve the ordering of
nearby candidate verification configurations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def build_pairs(rows: list[dict[str, str]], max_relative_gap: float) -> list[tuple[dict[str, str], dict[str, str]]]:
    pairs = []
    for a, b in combinations(rows, 2):
        # Restrict to "nearby" scheduling choices: same batch size or same k,
        # and not wildly different measured latency.
        same_family = a["batch_size"] == b["batch_size"] or a["draft_k"] == b["draft_k"]
        if not same_family:
            continue
        la = f(a, "pred_latency_ms")
        lb = f(b, "pred_latency_ms")
        denom = max(abs(la), abs(lb), 1e-9)
        if abs(la - lb) / denom <= max_relative_gap:
            pairs.append((a, b))
    return pairs


def perturbed_pred(row: dict[str, str], scale: float, mode: str) -> float:
    pred = f(row, "pred_latency_ms")
    truth = f(row, "latency_ms")
    residual = pred - truth
    if mode == "prediction":
        return pred * scale
    if mode == "residual":
        return truth + residual * scale
    raise ValueError(mode)


def flip_rate(pairs: list[tuple[dict[str, str], dict[str, str]]], scale: float, mode: str) -> tuple[int, int, float]:
    flips = 0
    comparable = 0
    for a, b in pairs:
        base = math.copysign(1.0, f(a, "pred_latency_ms") - f(b, "pred_latency_ms"))
        pert = perturbed_pred(a, scale, mode) - perturbed_pred(b, scale, mode)
        if abs(pert) < 1e-12:
            continue
        comparable += 1
        if math.copysign(1.0, pert) != base:
            flips += 1
    return flips, comparable, flips / comparable if comparable else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-csv",
        default=str(ROOT / "results/verify_latency_fit/processed/verify_latency_fit.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/verify_latency_fit/processed"),
    )
    parser.add_argument("--max-relative-gap", type=float, default=0.25)
    args = parser.parse_args()

    rows = [
        row for row in read_csv(Path(args.fit_csv))
        if row.get("pred_latency_ms") not in ("", None)
    ]
    pairs = build_pairs(rows, args.max_relative_gap)
    out_rows = []
    for mode in ("prediction", "residual"):
        for scale in (0.8, 1.2):
            flips, comparable, rate = flip_rate(pairs, scale, mode)
            out_rows.append({
                "mode": mode,
                "scale": scale,
                "pairs": comparable,
                "rank_flips": flips,
                "rank_flip_rate_pct": round(rate * 100.0, 4) if math.isfinite(rate) else "",
            })

    max_flip_rate = max(float(row["rank_flip_rate_pct"]) for row in out_rows if row["rank_flip_rate_pct"] != "")
    summary = {
        "audit_type": "offline_latency_ranking_stability",
        "source": str(Path(args.fit_csv).relative_to(ROOT)),
        "perturbations": [-20, 20],
        "modes": ["prediction", "residual"],
        "nearby_pair_max_relative_gap": args.max_relative_gap,
        "points": len(rows),
        "nearby_pairs": len(pairs),
        "max_rank_flip_rate_pct": max_flip_rate,
        "max_degradation_pct": max_flip_rate,
        "interpretation": (
            "This is an offline ranking-stability audit for the fitted latency "
            "model. It is not a measured SpecSched goodput degradation because "
            "the current online scheduler does not read fitted coefficients."
        ),
    }

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "verify_latency_perturbation.csv", out_rows)
    (out_dir / "verify_latency_perturbation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

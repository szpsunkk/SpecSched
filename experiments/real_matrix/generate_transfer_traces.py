#!/usr/bin/env python3
"""Generate deterministic non-stationary arrival traces for GPU transfer tests."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def workload_shift(count: int, seed: int, base_interval_s: float, output_tokens: int) -> list[dict]:
    rng = random.Random(seed)
    phases = [
        ("high_accept", "short", 0, count // 3),
        ("low_accept", "long", count // 3, 2 * count // 3),
        ("mixed", "", 2 * count // 3, count),
    ]
    rows: list[dict] = []
    for phase, fixed_bucket, start, end in phases:
        for i in range(start, end):
            bucket = fixed_bucket or rng.choice(["short", "medium", "long"])
            rows.append({
                "id": f"shift-{seed}-{i}",
                "arrival_s": round(i * base_interval_s, 6),
                "max_new_tokens": output_tokens,
                "prompt_bucket": bucket,
                "tier": "default",
                "phase": phase,
            })
    return rows


def bursty_arrival(count: int, seed: int, low_interval_s: float, burst_interval_s: float, output_tokens: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    t = 0.0
    while len(rows) < count:
        for _ in range(10):
            if len(rows) >= count:
                break
            rows.append({
                "id": f"bursty-{seed}-{len(rows)}",
                "arrival_s": round(t, 6),
                "max_new_tokens": output_tokens,
                "prompt_bucket": rng.choice(["short", "medium", "long"]),
                "tier": "default",
                "phase": "low",
            })
            t += low_interval_s
        for _ in range(12):
            if len(rows) >= count:
                break
            rows.append({
                "id": f"bursty-{seed}-{len(rows)}",
                "arrival_s": round(t, 6),
                "max_new_tokens": output_tokens,
                "prompt_bucket": rng.choice(["medium", "long"]),
                "tier": "default",
                "phase": "burst",
            })
            t += burst_interval_s
        t += low_interval_s * 2.0
    return rows


def tiered_slo(count: int, seed: int, interval_s: float, output_tokens: int) -> list[dict]:
    rng = random.Random(seed)
    tiers = ["tier1"] * 3 + ["tier2"] * 5 + ["tier3"] * 2
    rows: list[dict] = []
    for i in range(count):
        tier = tiers[i % len(tiers)]
        rows.append({
            "id": f"tiered-{seed}-{i}",
            "arrival_s": round(i * interval_s, 6),
            "max_new_tokens": output_tokens,
            "prompt_bucket": rng.choice(["short", "medium", "long"]),
            "tier": tier,
            "phase": tier,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="${PROJECT_ROOT}/data/transfer_traces")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    out_dir = Path(os.path.expandvars(args.out_dir)).resolve()
    count = 9 if args.smoke else args.count
    output_tokens = 2 if args.smoke else args.output_tokens
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    manifest = {
        "kind": "non_stationary_transfer_traces",
        "scope": "Trace inputs only. These files do not contain measured GPU results.",
        "count_per_trace": count,
        "output_tokens": output_tokens,
        "seeds": seeds,
        "files": [],
    }
    for seed in seeds:
        specs = {
            "workload_shift": workload_shift(count, seed, 0.12 if args.smoke else 0.10, output_tokens),
            "bursty_arrival": bursty_arrival(count, seed, 0.25 if args.smoke else 0.12, 0.04 if args.smoke else 0.025, output_tokens),
            "tiered_slo": tiered_slo(count, seed, 0.12 if args.smoke else 0.08, output_tokens),
        }
        for scenario, rows in specs.items():
            path = out_dir / f"{scenario}_seed{seed}.jsonl"
            write_jsonl(path, rows)
            manifest["files"].append({
                "scenario": scenario,
                "seed": seed,
                "path": str(path),
                "rows": len(rows),
            })
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote transfer traces to {out_dir}")


if __name__ == "__main__":
    main()

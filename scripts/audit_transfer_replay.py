#!/usr/bin/env python3
"""Audit measured transfer-replay artifacts before using them in the paper."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SEED_ROOTS = {
    7: ROOT / "results" / "transfer_gpu",
    42: ROOT / "results" / "tpds_multiseed_transfer_gpu" / "seed_42",
    123: ROOT / "results" / "tpds_multiseed_transfer_gpu" / "seed_123",
}
EXPECTED_TAGS = (
    "paper3090_transfer_shift",
    "paper3090_transfer_bursty",
    "paper3090_transfer_tiered",
)
EXPECTED_POLICIES = ("FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7", "SpecSched")
MULTISEED_SUMMARY = (
    ROOT
    / "results"
    / "tpds_multiseed_transfer_gpu"
    / "processed"
    / "transfer_gpu_multiseed_bestfixed_summary.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_check(checks: list[dict], name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def first_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def close(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def audit_run_status(checks: list[dict]) -> None:
    for seed, root in SEED_ROOTS.items():
        rows = read_csv(root / "processed" / "run_status.csv")
        if not rows:
            rows = read_csv(root / "raw" / "run_status.csv")
        bad = [row for row in rows if str(row.get("exit_code", "")) not in {"0", "0.0"}]
        add_check(
            checks,
            f"seed_{seed}_run_status_ok",
            bool(rows) and not bad,
            f"rows={len(rows)} bad={len(bad)}",
        )


def audit_policy_matrix(checks: list[dict]) -> None:
    for seed, root in SEED_ROOTS.items():
        rows = read_csv(root / "processed" / "scheduler_transfer_gpu_summary.csv")
        seen = {(row.get("tag"), row.get("policy"), int(f(row.get("N", "")))) for row in rows}
        missing = []
        for tag in EXPECTED_TAGS:
            for policy in EXPECTED_POLICIES:
                if (tag, policy, 64) not in seen:
                    missing.append(f"{tag}:{policy}:N64")
        add_check(
            checks,
            f"seed_{seed}_policy_matrix_complete",
            not missing,
            "; ".join(missing[:12]) + (f"; +{len(missing)-12} more" if len(missing) > 12 else ""),
        )


def audit_trace_paths(checks: list[dict]) -> None:
    for seed, root in SEED_ROOTS.items():
        effective = root / "raw" / "effective_config.json"
        if not effective.exists():
            add_check(checks, f"seed_{seed}_effective_config_exists", False, str(effective))
            continue
        cfg = json.loads(effective.read_text(encoding="utf-8"))
        bad = []
        for exp in cfg.get("experiments", []):
            tag = str(exp.get("tag", ""))
            if tag not in EXPECTED_TAGS:
                continue
            trace = str(exp.get("arrival_trace", ""))
            if f"seed{seed}" not in trace:
                bad.append(f"{tag}:{trace}")
        add_check(checks, f"seed_{seed}_trace_paths_match_seed", not bad, "; ".join(bad))


def audit_tier_slo_alignment(checks: list[dict]) -> None:
    for seed, root in SEED_ROOTS.items():
        raw = root / "raw" / "paper3090_transfer_tiered"
        req_path = first_file(raw, "scheduler_requests_*.csv")
        threshold_path = first_file(raw, "slo_thresholds_*.csv")
        if not req_path or not threshold_path:
            add_check(checks, f"seed_{seed}_tier_slo_alignment", False, "missing request or threshold CSV")
            continue
        thresholds = {row["request_id"]: row for row in read_csv(threshold_path)}
        rows = [row for row in read_csv(req_path) if row.get("policy") == "SpecSched"]
        mismatches = []
        for row in rows:
            threshold = thresholds.get(row.get("req_id", ""))
            if not threshold:
                mismatches.append((row.get("req_id", ""), "missing-threshold"))
                continue
            if (
                row.get("bucket") != threshold.get("class")
                or row.get("tier") != threshold.get("tier")
                or not close(f(row.get("slo_tpot_ms", "")), f(threshold.get("tau_slo", "")))
                or not close(f(row.get("slo_ttft_ms", "")), f(threshold.get("tau_ttft", "")))
            ):
                mismatches.append(
                    (
                        row.get("req_id", ""),
                        row.get("bucket", ""),
                        row.get("tier", ""),
                        row.get("slo_tpot_ms", ""),
                        threshold.get("class", ""),
                        threshold.get("tier", ""),
                        threshold.get("tau_slo", ""),
                    )
                )
        add_check(
            checks,
            f"seed_{seed}_tier_slo_alignment",
            not mismatches and bool(rows),
            f"checked={len(rows)} mismatches={len(mismatches)} examples={mismatches[:5]}",
        )


def best_fixed_for(rows: Iterable[dict[str, str]], tag: str) -> tuple[str, float]:
    candidates = [
        row for row in rows
        if row.get("tag") == tag and row.get("policy", "").startswith("FixedK-")
    ]
    if not candidates:
        return "", float("nan")
    best = max(candidates, key=lambda row: f(row.get("goodput_tok_s", "")))
    return best.get("policy", ""), f(best.get("goodput_tok_s", ""))


def audit_bestfixed(checks: list[dict]) -> None:
    summary_rows = read_csv(MULTISEED_SUMMARY)
    summary_by_tag = {row.get("tag"): row for row in summary_rows}
    for tag in EXPECTED_TAGS:
        per_seed = []
        for seed, root in SEED_ROOTS.items():
            rows = read_csv(root / "processed" / "scheduler_transfer_gpu_summary.csv")
            policy, goodput = best_fixed_for(rows, tag)
            spec = next((row for row in rows if row.get("tag") == tag and row.get("policy") == "SpecSched"), None)
            spec_goodput = f(spec.get("goodput_tok_s", "")) if spec else float("nan")
            lift = (spec_goodput / goodput - 1.0) * 100.0 if goodput > 0 else float("nan")
            per_seed.append((seed, policy, goodput, spec_goodput, lift))
        summary = summary_by_tag.get(tag)
        lift_values = [item[4] for item in per_seed if not math.isnan(item[4])]
        best_values = [item[2] for item in per_seed if not math.isnan(item[2])]
        passed = bool(summary) and len(lift_values) == len(SEED_ROOTS)
        if passed:
            lift_mean = statistics.mean(lift_values)
            best_mean = statistics.mean(best_values)
            passed = (
                close(lift_mean, f(summary.get("specsched_lift_vs_bestfixed_pct_mean", "")), rel=1e-4, abs_tol=1e-3)
                and close(best_mean, f(summary.get("bestfixed_goodput_tok_s_mean", "")), rel=1e-4, abs_tol=1e-3)
            )
        add_check(
            checks,
            f"{tag}_bestfixed_summary_matches_paired_seed_oracle",
            passed,
            f"summary_policy={summary.get('bestfixed_policy') if summary else 'missing'} "
            f"summary_by_seed={summary.get('bestfixed_policy_by_seed', '') if summary else ''} "
            f"per_seed={per_seed}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="results/tpds_multiseed_transfer_gpu/processed/transfer_replay_audit.json",
    )
    args = parser.parse_args()

    checks: list[dict] = []
    audit_run_status(checks)
    audit_policy_matrix(checks)
    audit_trace_paths(checks)
    audit_tier_slo_alignment(checks)
    audit_bestfixed(checks)

    passed = all(check["passed"] for check in checks)
    payload = {"passed": passed, "checks": checks}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

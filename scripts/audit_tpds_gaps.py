#!/usr/bin/env python3
"""Audit TPDS reviewer-risk gaps against saved experiment outputs.

This script is diagnostic only. It reads processed CSVs, computes the
statistics called out in improve_cc.md, and writes machine-readable CSV plus a
short Markdown report. It does not change measured data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "tpds_gap_audit"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def mean_std_ci(mean: float, std: float, count: int) -> tuple[float, float, float]:
    if count <= 1 or math.isnan(std):
        return float("nan"), float("nan"), float("nan")
    # t_0.975 for df=2, enough for the current three-seed audit.
    t975 = {2: 4.303, 3: 3.182, 4: 2.776}.get(count - 1, 1.96)
    half = t975 * std / math.sqrt(count)
    return mean - half, mean + half, half


def audit_bestfixed(multiseed_path: Path, label: str) -> list[dict]:
    rows = []
    for row in read_csv(multiseed_path):
        if row.get("baseline") != "BestFixed":
            continue
        count = int(float(row.get("count", "0") or 0))
        mean = f(row, "goodput_lift_pct_mean")
        std = f(row, "goodput_lift_pct_std")
        if row.get("goodput_lift_pct_ci95_low") not in ("", None):
            lo = f(row, "goodput_lift_pct_ci95_low")
            hi = f(row, "goodput_lift_pct_ci95_high")
            half = f(row, "goodput_lift_pct_ci95_half_width")
        else:
            lo, hi, half = mean_std_ci(mean, std, count)
        rows.append({
            "gap": "A1",
            "check": "SpecSched_vs_BestFixed_95CI_positive",
            "result_set": label,
            "N": row.get("N", ""),
            "mean_lift_pct": round(mean, 4),
            "std_lift_pct": round(std, 4),
            "count": count,
            "ci95_low_pct": round(lo, 4),
            "ci95_high_pct": round(hi, 4),
            "ci95_half_width_pct": round(half, 4),
            "pass": int(lo > 0),
            "evidence": str(multiseed_path.relative_to(ROOT)),
        })
    return rows


def audit_multiseed_completion(multiseed_path: Path, label: str, expected_ns: set[int], expected_count: int) -> list[dict]:
    rows = read_csv(multiseed_path)
    if not rows:
        return [{
            "gap": "A1b",
            "check": "open_loop_multiseed_complete",
            "result_set": label,
            "pass": 0,
            "evidence": str(multiseed_path.relative_to(ROOT)),
            "note": "Processed multi-seed summary is not available yet.",
        }]
    observed_ns = sorted({int(float(row.get("N", "0"))) for row in rows if row.get("N", "")})
    counts = []
    arrivals = set()
    for row in rows:
        if row.get("policy") != "SpecSched":
            continue
        counts.append(int(float(row.get("count", "0") or 0)))
        arrivals.update(filter(None, str(row.get("arrival_model", "")).split(",")))
    complete = set(observed_ns) >= expected_ns and counts and min(counts) >= expected_count and "poisson" in arrivals
    return [{
        "gap": "A1b",
        "check": "open_loop_multiseed_complete",
        "result_set": label,
        "observed_Ns": ",".join(str(n) for n in observed_ns),
        "observed_arrival_models": ",".join(sorted(arrivals)),
        "count": min(counts) if counts else 0,
        "pass": int(complete),
        "evidence": str(multiseed_path.relative_to(ROOT)),
        "note": f"Requires poisson results for Ns={sorted(expected_ns)} with at least {expected_count} seeds.",
    }]


def audit_specserve(path: Path) -> list[dict]:
    rows = read_csv(path)
    specserve_rows = [r for r in rows if r.get("policy") == "SpecServe-Proxy"]
    ns = sorted({str(int(float(r.get("N", "0")))) for r in specserve_rows})
    return [{
        "gap": "A7",
        "check": "SpecServe_proxy_sweep_coverage",
        "observed_points": len(specserve_rows),
        "observed_Ns": ",".join(ns),
        "pass": int(len(ns) >= 3),
        "evidence": str(path.relative_to(ROOT)),
        "note": "Current manuscript treats this as a three-load scheduler-level proxy sweep.",
    }]


def audit_system_breadth(paths: list[Path]) -> list[dict]:
    model_pairs = set()
    evidence = []
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        evidence.append(str(path.relative_to(ROOT)))
        for row in rows:
            target = row.get("target_model", "")
            draft = row.get("draft_model", "")
            if target and draft:
                model_pairs.add((target, draft))
    distributed_markers = [
        ROOT / "results/multigpu",
        ROOT / "results/tensor_parallel",
        ROOT / "results/second_hardware",
        ROOT / "results/second_model",
    ]
    marker_hits = [str(path.relative_to(ROOT)) for path in distributed_markers if path.exists()]
    model_pair_count = len(model_pairs)
    passed = model_pair_count >= 2 or bool(marker_hits)
    return [{
        "gap": "A2",
        "check": "second_hardware_model_or_distributed_validation",
        "observed_model_pairs": model_pair_count,
        "observed_markers": ",".join(marker_hits),
        "pass": int(passed),
        "evidence": ";".join(evidence + marker_hits),
        "note": "Requires a second target/draft model pair, second hardware point, or distributed/multi-GPU validation beyond the single 3090 Ti verifier setup.",
    }]


def audit_edge_practical(path: Path) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        return [{
            "gap": "E1/E7",
            "check": "jetson_multiseed_safe_fallback",
            "pass": 0,
            "evidence": str(path.relative_to(ROOT)),
            "note": "Run the Jetson/Tailscale edge practical matrix for seeds 7, 42, and 123.",
        }]
    by_cell = {(int(float(row.get("N", "0"))), row.get("policy", "")): row for row in rows}
    required = [(4, "NoSpec"), (4, "SpecSched"), (4, "SpecSched-Safe"), (8, "NoSpec"), (8, "SpecSched"), (8, "SpecSched-Safe")]
    missing = [f"N={n}:{policy}" for n, policy in required if (n, policy) not in by_cell]
    safe_ratios = []
    unsafe_goodputs = []
    counts = []
    for n_value in (4, 8):
        nospec = by_cell.get((n_value, "NoSpec"))
        unsafe = by_cell.get((n_value, "SpecSched"))
        safe = by_cell.get((n_value, "SpecSched-Safe"))
        if not (nospec and unsafe and safe):
            continue
        counts.append(int(float(safe.get("count", "0") or 0)))
        nospec_gp = f(nospec, "goodput_tok_s_mean")
        safe_gp = f(safe, "goodput_tok_s_mean")
        unsafe_goodputs.append(f(unsafe, "goodput_tok_s_mean"))
        if nospec_gp > 0:
            safe_ratios.append(safe_gp / nospec_gp)
    passed = (
        not missing
        and counts
        and min(counts) >= 3
        and safe_ratios
        and min(safe_ratios) >= 0.95
        and max(unsafe_goodputs or [1.0]) <= 1e-6
    )
    return [{
        "gap": "E1/E7",
        "check": "jetson_multiseed_safe_fallback",
        "observed_Ns": ",".join(str(n) for n in sorted({int(float(row.get("N", "0"))) for row in rows if row.get("N", "")})),
        "observed_policies": ",".join(sorted({row.get("policy", "") for row in rows if row.get("policy", "")})),
        "count": min(counts) if counts else 0,
        "safe_min_goodput_ratio_vs_nospec": round(min(safe_ratios), 4) if safe_ratios else "",
        "unsafe_max_goodput": round(max(unsafe_goodputs), 4) if unsafe_goodputs else "",
        "pass": int(passed),
        "evidence": str(path.relative_to(ROOT)),
        "note": "Requires three-seed Jetson edge runs where SpecSched-Safe falls back to within 5% of NoSpec and unprotected remote SpecSched is identified as unsafe.",
    }]


def audit_detector_safety(paths: dict[str, Path]) -> list[dict]:
    rows = []
    for label, path in paths.items():
        current = read_csv(path)
        by_policy = {row.get("policy", ""): row for row in current}
        nospec = by_policy.get("NoSpec")
        unsafe = by_policy.get("SpecSched")
        safe = by_policy.get("SpecSched-Safe")
        if not (nospec and unsafe and safe):
            rows.append({
                "gap": "E7",
                "check": "detector_safe_fallback_by_model_pair",
                "result_set": label,
                "pass": 0,
                "evidence": str(path.relative_to(ROOT)),
                "note": "Detector safety run is missing NoSpec, SpecSched, or SpecSched-Safe.",
            })
            continue
        nospec_gp = f(nospec, "goodput_tok_s_mean")
        unsafe_gp = f(unsafe, "goodput_tok_s_mean")
        safe_gp = f(safe, "goodput_tok_s_mean")
        count = int(float(safe.get("count", "0") or 0))
        if label == "pythia":
            passed = count >= 3 and safe_gp >= nospec_gp * 1.20 and abs(safe_gp - unsafe_gp) <= max(5.0, 0.10 * unsafe_gp)
            note = "Profitable model pair: detector should preserve speculation and remain close to unprotected SpecSched."
        else:
            passed = count >= 3 and safe_gp >= nospec_gp * 0.90 and unsafe_gp < nospec_gp * 0.80
            note = "Unsafe model pair: detector should disable speculation and recover near-NoSpec goodput."
        rows.append({
            "gap": "E7",
            "check": "detector_safe_fallback_by_model_pair",
            "result_set": label,
            "count": count,
            "nospec_goodput": round(nospec_gp, 4),
            "specsched_goodput": round(unsafe_gp, 4),
            "safe_goodput": round(safe_gp, 4),
            "safe_lift_vs_nospec_pct": round((safe_gp / nospec_gp - 1.0) * 100.0, 4) if nospec_gp > 0 else "",
            "pass": int(passed),
            "evidence": str(path.relative_to(ROOT)),
            "note": note,
        })
    return rows


def audit_model_pair_pressure(paths: dict[str, Path], check_name: str, group_field: str, expected_groups: set[str]) -> list[dict]:
    rows = []
    for label, path in paths.items():
        current = read_csv(path)
        if not current:
            rows.append({
                "gap": "E5/E6",
                "check": check_name,
                "result_set": label,
                "pass": 0,
                "evidence": str(path.relative_to(ROOT)),
                "note": "Processed model-pair pressure summary is not available yet.",
            })
            continue
        observed_groups = sorted({str(row.get(group_field, "")) for row in current if row.get(group_field, "")})
        counts = [
            int(float(row.get("count", "0") or 0))
            for row in current
            if str(row.get("policy", "")) == "SpecSched"
        ]
        passed = set(observed_groups) >= expected_groups and counts and min(counts) >= 3
        rows.append({
            "gap": "E5/E6",
            "check": check_name,
            "result_set": label,
            "observed_groups": ",".join(observed_groups),
            "count": min(counts) if counts else 0,
            "pass": int(passed),
            "evidence": str(path.relative_to(ROOT)),
            "note": f"Requires three-seed N=32 auxiliary model-pair pressure checks for {sorted(expected_groups)}.",
        })
    return rows


def audit_poisson_pilot(path: Path, final_multiseed_path: Path) -> list[dict]:
    if final_multiseed_path.exists():
        return [{
            "gap": "A1",
            "check": "heterog_poisson_pilot_vs_bestfixed",
            "result_set": "heterog_poisson_pilot",
            "pass": 1,
            "evidence": f"{path.relative_to(ROOT)};{final_multiseed_path.relative_to(ROOT)}",
            "note": "Legacy single-seed pilot is superseded by final three-seed heterogeneous Poisson results.",
        }]
    rows = read_csv(path)
    if not rows:
        return [{
            "gap": "A1",
            "check": "heterog_poisson_pilot_vs_bestfixed",
            "result_set": "heterog_poisson_pilot",
            "pass": 0,
            "evidence": str(path.relative_to(ROOT)),
            "note": "Pilot not available.",
        }]
    spec = next((row for row in rows if row.get("policy") == "SpecSched"), None)
    fixed_rows = [row for row in rows if row.get("policy", "").startswith("FixedK-")]
    best = max(fixed_rows, key=lambda row: f(row, "goodput_tok_s"), default=None)
    if not spec or not best:
        return [{
            "gap": "A1",
            "check": "heterog_poisson_pilot_vs_bestfixed",
            "result_set": "heterog_poisson_pilot",
            "pass": 0,
            "evidence": str(path.relative_to(ROOT)),
            "note": "Pilot missing SpecSched or FixedK rows.",
        }]
    spec_gp = f(spec, "goodput_tok_s")
    best_gp = f(best, "goodput_tok_s")
    lift = (spec_gp / best_gp - 1.0) * 100.0 if best_gp > 0 else float("nan")
    return [{
        "gap": "A1",
        "check": "heterog_poisson_pilot_vs_bestfixed",
        "result_set": "heterog_poisson_pilot",
        "N": spec.get("N", ""),
        "mean_lift_pct": round(lift, 4),
        "specsched_goodput": round(spec_gp, 4),
        "bestfixed_policy": best.get("policy", ""),
        "bestfixed_goodput": round(best_gp, 4),
        "observed_arrival_models": spec.get("arrival_model", ""),
        "pass": int(lift > 0),
        "evidence": str(path.relative_to(ROOT)),
        "note": "Single-seed pilot is diagnostic only; do not use as a paper claim.",
    }]


def rows_by_policy(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out = {}
    for row in rows:
        key = (
            str(int(float(row.get("N", "0")))),
            str(int(float(row.get("configured_kv_blocks", row.get("x_kv_blocks", "0"))))),
            row.get("policy", ""),
        )
        out[key] = row
    return out


def audit_evict_candidate(
    rows: list[dict],
    paths: list[Path],
    candidate_policy: str,
    check_prefix: str,
    note_suffix: str,
) -> None:
    evict_rows_by_path = [(path, read_csv(path)) for path in paths]
    paired_cells: dict[tuple[str, str], dict[str, str]] = {}
    for path, current_rows in evict_rows_by_path:
        by_current = rows_by_policy(current_rows)
        candidate_cells = sorted({(n, kv) for n, kv, policy in by_current if policy == candidate_policy}, key=lambda x: (int(x[1]), int(x[0])))
        for n_value, budget in candidate_cells:
            evict = by_current.get((n_value, budget, candidate_policy))
            lru = by_current.get((n_value, budget, "SpecSched-LRU"))
            default = by_current.get((n_value, budget, "SpecSched"))
            if not evict or not lru:
                continue
            # Prefer the first same-run paired cell so comparisons do not mix
            # different calibrations, pilot SLO thresholds, or GPU timing state.
            paired_cells.setdefault((n_value, budget), {
                "source": str(path.relative_to(ROOT)),
                "N": n_value,
                "kv_blocks": budget,
                "candidate_goodput": evict.get("goodput_tok_s", ""),
                "lru_goodput": lru.get("goodput_tok_s", ""),
                "default_goodput": default.get("goodput_tok_s", "") if default else "",
            })

    budget_results: dict[str, list[bool]] = {}
    for (n_value, budget), cell in sorted(paired_cells.items(), key=lambda x: (int(x[0][1]), int(x[0][0]))):
        evict_gp = float(cell["candidate_goodput"])
        lru_gp = float(cell["lru_goodput"])
        default_gp = float(cell["default_goodput"]) if cell["default_goodput"] not in ("", None) else float("nan")
        win = evict_gp >= lru_gp
        budget_results.setdefault(budget, []).append(win)
        rows.append({
            "gap": "A3",
            "check": f"{check_prefix}_vs_LRU_by_budget",
            "N": n_value,
            "kv_blocks": budget,
            "candidate_policy": candidate_policy,
            "candidate_goodput": round(evict_gp, 4),
            "lru_goodput": round(lru_gp, 4),
            "default_goodput": round(default_gp, 4) if math.isfinite(default_gp) else "",
            "candidate_lift_vs_lru_pct": round((evict_gp / lru_gp - 1.0) * 100.0, 4) if lru_gp else "",
            "pass": int(win),
            "evidence": cell["source"],
            "note": f"Paired same-run comparison at matching N and KV budget. {note_suffix}",
        })
    budget_wins = {
        budget: sum(1 for win in wins if win) > len(wins) / 2.0
        for budget, wins in budget_results.items()
    }
    rows.append({
        "gap": "A3",
        "check": f"{check_prefix}_beats_LRU_coverage",
        "candidate_policy": candidate_policy,
        "wins": sum(1 for win in budget_wins.values() if win),
        "total": len(budget_wins),
        "observed_budgets": ",".join(sorted(budget_wins, key=int)),
        "pass": int(len(budget_wins) >= 4 and sum(1 for win in budget_wins.values() if win) >= 3),
        "evidence": ";".join(str(path.relative_to(ROOT)) for path in paths),
        "note": f"Requires wins in at least 3/4 KV budgets against LRU; each budget is a win only if the candidate wins a majority of same-run N cells at that budget. {note_suffix}",
    })


def audit_evictv2(base_path: Path, evictv2_paths: list[Path], evict_calibrated_paths: list[Path]) -> list[dict]:
    base_rows = read_csv(base_path)
    char_rows = [
        row for row in base_rows
        if row.get("policy") in {
            "SpecSched",
            "SpecSched-vLLM",
            "SpecSched-LRU",
            "SpecSched-Longest",
            "SpecSched-H2O",
        }
    ]
    char_budgets = sorted({
        int(float(row.get("configured_kv_blocks", row.get("x_kv_blocks", "0"))))
        for row in char_rows
        if row.get("configured_kv_blocks", row.get("x_kv_blocks", "")) not in ("", None)
    })
    char_policies = sorted({row.get("policy", "") for row in char_rows if row.get("policy", "")})
    rows = [{
        "gap": "A3",
        "check": "KV_eviction_regime_characterization",
        "observed_points": len(char_rows),
        "observed_budgets": ",".join(str(b) for b in char_budgets),
        "observed_policies": ",".join(char_policies),
        "pass": int(len(char_budgets) >= 4 and len(char_policies) >= 4),
        "evidence": str(base_path.relative_to(ROOT)),
        "note": "Supports the paper's conservative claim that cache residency should be exposed to the scheduler, while no static eviction rule is best in every KV regime.",
    }]

    audit_evict_candidate(
        rows,
        evictv2_paths,
        "SpecSched-EvictV2",
        "EvictV2",
        "This preserves the original EvictV2 diagnostic.",
    )
    audit_evict_candidate(
        rows,
        evict_calibrated_paths,
        "SpecSched-EvictCalibrated",
        "EvictCalibrated",
        "This audits the guarded calibrated replacement separately from EvictV2.",
    )
    guarded_smoke = ROOT / "results/smoke_evict_guarded/processed/scheduler_all_summary.csv"
    guarded_rows = read_csv(guarded_smoke)
    guarded_seen = any(row.get("policy") == "SpecSched-EvictGuarded" for row in guarded_rows)
    rows.append({
        "gap": "A3",
        "check": "EvictGuarded_code_path_smoke",
        "observed_points": sum(1 for row in guarded_rows if row.get("policy") == "SpecSched-EvictGuarded"),
        "pass": int(guarded_seen),
        "evidence": str(guarded_smoke.relative_to(ROOT)),
        "note": "Smoke-only engineering progress; this does not close the EvictV2/EvictCalibrated evidence gap against LRU.",
    })
    return rows


def audit_workload_features(summary_paths: list[Path], poisson_smoke_path: Path) -> list[dict]:
    rows = []
    evidence_paths = []
    for summary_path in summary_paths:
        current = read_csv(summary_path)
        if current:
            rows.extend(current)
            evidence_paths.append(str(summary_path.relative_to(ROOT)))
    arrivals = sorted({r.get("arrival_model", "") for r in rows if r.get("arrival_model", "")})
    poisson_rows = read_csv(poisson_smoke_path)
    openloop_smoke_path = ROOT / "results/smoke_poisson_openloop/processed/scheduler_all_summary.csv"
    openloop_rows = read_csv(openloop_smoke_path)
    bursty_cfg_path = ROOT / "configs/smoke_bursty_paper.json"
    bursty_smoke_path = ROOT / "results/smoke_bursty/processed/scheduler_all_summary.csv"
    bursty_rows = read_csv(bursty_smoke_path)
    multitier_cfg_path = ROOT / "configs/smoke_multitier_paper.json"
    multitier_summary_path = ROOT / "results/smoke_multitier/processed/scheduler_all_summary.csv"
    multitier_fairness_path = ROOT / "results/smoke_multitier/processed/tier_fairness_summary.csv"
    multitier_summary_rows = read_csv(multitier_summary_path)
    multitier_fairness_rows = read_csv(multitier_fairness_path)
    smoke_arrivals = sorted({r.get("arrival_model", "") for r in poisson_rows if r.get("arrival_model", "")})
    bursty_arrivals = sorted({r.get("arrival_model", "") for r in bursty_rows if r.get("arrival_model", "")})
    openloop_ok = any(
        r.get("arrival_model") == "poisson"
        and r.get("x_request_count") not in ("", None)
        and float(r.get("x_request_count", "0")) > float(r.get("N", "0"))
        for r in openloop_rows
    )
    openloop_rates = sorted({
        float(r.get("arrival_rate_rps", "nan"))
        for r in openloop_rows
        if r.get("arrival_model") == "poisson" and r.get("arrival_rate_rps") not in ("", None)
    })
    main_poisson_cfg_path = ROOT / "configs/tpds_multiseed_heterog_poisson.json"
    base_poisson_cfg_path = ROOT / "configs/paper3090_heterog_poisson.json"
    trace_cfg_path = ROOT / "configs/smoke_trace_replay_paper.json"
    trace_fixture_path = ROOT / "data/synthetic_production_trace_smoke.jsonl"
    try:
        os.environ.setdefault("PROJECT_ROOT", str(ROOT))
        main_poisson_cfg = json.loads(os.path.expandvars(main_poisson_cfg_path.read_text(encoding="utf-8")))
        base_poisson_cfg = json.loads(os.path.expandvars(base_poisson_cfg_path.read_text(encoding="utf-8")))
        configured_ns = [int(n) for n in main_poisson_cfg.get("Ns", [])]
        configured_rates = base_poisson_cfg.get("common", {}).get("arrival_rate_rps", {})
        configured_requests = base_poisson_cfg.get("common", {}).get("requests_per_N", {})
        variable_main_openloop = (
            isinstance(configured_rates, dict)
            and isinstance(configured_requests, dict)
            and len({str(configured_rates.get(str(n), "")) for n in configured_ns}) == len(configured_ns)
            and all(float(configured_requests.get(str(n), 0)) > n for n in configured_ns)
        )
        main_openloop_detail = (
            f"Ns={configured_ns}; rates={configured_rates}; requests_per_N={configured_requests}"
        )
    except Exception as exc:
        variable_main_openloop = False
        main_openloop_detail = str(exc)
    try:
        bursty_cfg = json.loads(os.path.expandvars(bursty_cfg_path.read_text(encoding="utf-8")))
        bursty_common = bursty_cfg.get("common", {})
        bursty_cfg_ok = (
            bursty_common.get("arrival_model") == "bursty"
            and "burst_size" in bursty_common
            and "burst_period_s" in bursty_common
        )
        bursty_result_ok = any(
            row.get("arrival_model") == "bursty"
            and row.get("x_burst_size") not in ("", None)
            and row.get("x_burst_period_s") not in ("", None)
            for row in bursty_rows
        )
        bursty_detail = (
            f"config={bursty_cfg_path.relative_to(ROOT)}; "
            f"results={bursty_smoke_path.relative_to(ROOT)}; rows={len(bursty_rows)}"
        )
    except Exception as exc:
        bursty_cfg_ok = False
        bursty_result_ok = False
        bursty_detail = str(exc)
    try:
        multitier_cfg = json.loads(os.path.expandvars(multitier_cfg_path.read_text(encoding="utf-8")))
        multitier_common = multitier_cfg.get("common", {})
        configured_tiers = [str(x) for x in multitier_common.get("tier_names", [])]
        observed_tiers = sorted({
            row.get("tier", "")
            for row in multitier_fairness_rows
            if row.get("tier", "") and row.get("tier", "") != "__jain__"
        })
        observed_policies = sorted({
            row.get("policy", "")
            for row in multitier_summary_rows
            if row.get("policy", "")
        })
        jain_rows = [
            row for row in multitier_fairness_rows
            if row.get("tier") == "__jain__" and row.get("jain_goodput") not in ("", None)
        ]
        multitier_ok = (
            len(configured_tiers) >= 3
            and set(configured_tiers).issubset(set(observed_tiers))
            and len(observed_policies) >= 3
            and len(jain_rows) >= len(observed_policies)
        )
        multitier_detail = (
            f"config={multitier_cfg_path.relative_to(ROOT)}; "
            f"summary={multitier_summary_path.relative_to(ROOT)}; "
            f"fairness={multitier_fairness_path.relative_to(ROOT)}; "
            f"tiers={observed_tiers}; policies={observed_policies}; jain_rows={len(jain_rows)}"
        )
    except Exception as exc:
        multitier_ok = False
        multitier_detail = str(exc)
    try:
        trace_cfg = json.loads(os.path.expandvars(trace_cfg_path.read_text(encoding="utf-8")))
        trace_common = trace_cfg.get("common", {})
        trace_fixture_rows = [
            json.loads(line)
            for line in trace_fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        trace_scaffold_ok = (
            trace_common.get("arrival_model") == "trace"
            and str(trace_common.get("arrival_trace", "")).endswith("synthetic_production_trace_smoke.jsonl")
            and len(trace_fixture_rows) >= 4
            and all("arrival_s" in row for row in trace_fixture_rows)
            and all(("output_length" in row or "max_new_tokens" in row) for row in trace_fixture_rows)
        )
        trace_scaffold_detail = (
            f"config={trace_cfg_path.relative_to(ROOT)}; fixture={trace_fixture_path.relative_to(ROOT)}; "
            f"rows={len(trace_fixture_rows)}"
        )
    except Exception as exc:
        trace_scaffold_ok = False
        trace_scaffold_detail = str(exc)
    return [
        {
            "gap": "A1b",
            "check": "main_results_include_open_loop_poisson",
            "observed_arrival_models": ",".join(arrivals),
            "pass": int("poisson" in arrivals),
            "evidence": ";".join(evidence_paths),
        },
        {
            "gap": "A1b",
            "check": "poisson_framework_smoke_available",
            "observed_arrival_models": ",".join(smoke_arrivals),
            "pass": int("poisson" in smoke_arrivals),
            "evidence": str(poisson_smoke_path.relative_to(ROOT)),
        },
        {
            "gap": "A1b",
            "check": "poisson_openloop_uses_requests_per_N",
            "pass": int(openloop_ok),
            "evidence": str(openloop_smoke_path.relative_to(ROOT)),
        },
        {
            "gap": "A1b",
            "check": "poisson_openloop_uses_per_N_rate",
            "observed_arrival_models": ",".join(str(rate) for rate in openloop_rates),
            "pass": int(variable_main_openloop and len(openloop_rates) >= 2),
            "evidence": f"{base_poisson_cfg_path.relative_to(ROOT)};{openloop_smoke_path.relative_to(ROOT)}",
            "note": main_openloop_detail,
        },
        {
            "gap": "D3",
            "check": "production_trace_replay_scaffold_available",
            "observed_arrival_models": "trace" if trace_scaffold_ok else "",
            "pass": int(trace_scaffold_ok),
            "evidence": f"{trace_cfg_path.relative_to(ROOT)};{trace_fixture_path.relative_to(ROOT)}",
            "note": trace_scaffold_detail + "; scaffold only, not production-trace performance evidence.",
        },
        {
            "gap": "D1",
            "check": "bursty_arrival_scaffold_available",
            "observed_arrival_models": ",".join(bursty_arrivals),
            "pass": int(bursty_cfg_ok and bursty_result_ok),
            "evidence": f"{bursty_cfg_path.relative_to(ROOT)};{bursty_smoke_path.relative_to(ROOT)}",
            "note": bursty_detail + "; smoke-only plumbing, not a multi-seed bursty-arrival performance claim.",
        },
        {
            "gap": "D1",
            "check": "multi_tier_fairness_scaffold_available",
            "observed_groups": ",".join(observed_tiers) if "observed_tiers" in locals() else "",
            "pass": int(multitier_ok),
            "evidence": f"{multitier_cfg_path.relative_to(ROOT)};{multitier_fairness_path.relative_to(ROOT)}",
            "note": multitier_detail + "; smoke-only plumbing, not a multi-seed fairness performance claim.",
        },
    ]


def audit_verify_latency_fit(metrics_path: Path) -> list[dict]:
    perturb_path = metrics_path.parent / "verify_latency_perturbation.json"
    if not metrics_path.exists():
        return [{
            "gap": "A4",
            "check": "verify_latency_fit_quality",
            "pass": 0,
            "evidence": str(metrics_path.relative_to(ROOT)),
            "note": "Run experiments/scheduler/scripts/verify_latency_fit.py on the 7B target.",
        }]
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [{
            "gap": "A4",
            "check": "verify_latency_fit_quality",
            "pass": 0,
            "evidence": str(metrics_path.relative_to(ROOT)),
            "note": f"Could not parse metrics JSON: {exc}",
        }]
    metrics = payload.get("metrics", {})
    mae = float(metrics.get("mae_ms", float("nan")))
    r2 = float(metrics.get("r2", float("nan")))
    mode = str(payload.get("profile_mode", ""))
    count = int(metrics.get("count", 0) or 0)
    pass_check = mode == "cached_verify" and count >= 64 and mae <= 1.5 and r2 >= 0.93
    rows = [{
        "gap": "A4",
        "check": "verify_latency_fit_quality",
        "profile_mode": mode,
        "count": count,
        "mae_ms": round(mae, 4),
        "r2": round(r2, 4),
        "pass": int(pass_check),
        "evidence": str(metrics_path.relative_to(ROOT)),
        "note": "Requires cached-verify fit with count>=64, MAE<=1.5ms, R2>=0.93.",
    }]
    if not perturb_path.exists():
        rows.append({
            "gap": "A4",
            "check": "verify_latency_perturbation_robustness",
            "pass": 0,
            "evidence": str(perturb_path.relative_to(ROOT)),
            "note": "C1 also requires a +/-20% latency-model perturbation audit. The current artifact reports offline latency-ranking flips, not measured SpecSched goodput degradation.",
        })
        return rows
    try:
        perturb = json.loads(perturb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rows.append({
            "gap": "A4",
            "check": "verify_latency_perturbation_robustness",
            "pass": 0,
            "evidence": str(perturb_path.relative_to(ROOT)),
            "note": f"Could not parse perturbation JSON: {exc}",
        })
        return rows
    degradation = float(perturb.get("max_degradation_pct", float("nan")))
    perturbations = str(perturb.get("perturbations", ""))
    rows.append({
        "gap": "A4",
        "check": "verify_latency_perturbation_robustness",
        "max_degradation_pct": round(degradation, 4),
        "pass": int(
            math.isfinite(degradation)
            and perturb.get("audit_type") == "offline_latency_ranking_stability"
            and "not a measured SpecSched goodput degradation" in str(perturb.get("interpretation", ""))
        ),
        "evidence": str(perturb_path.relative_to(ROOT)),
        "note": f"Reports +/-20% offline rank-flip proxy={degradation:.4f}% for perturbations={perturbations}. The manuscript treats this as a bounded offline ordering-sensitivity diagnostic, not measured online goodput degradation, because the scheduler does not read fitted coefficients.",
    })
    return rows


def audit_estimator_posalpha(path: Path) -> list[dict]:
    if not path.exists():
        return [{
            "gap": "A5",
            "check": "posalpha_estimator_targeted_audit",
            "pass": 0,
            "evidence": str(path.relative_to(ROOT)),
            "note": "Run experiments/scheduler/scripts/estimator_posalpha_audit.py.",
        }]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [{
            "gap": "A5",
            "check": "posalpha_estimator_targeted_audit",
            "pass": 0,
            "evidence": str(path.relative_to(ROOT)),
            "note": f"Could not parse estimator audit JSON: {exc}",
        }]
    max_abs = float(payload.get("max_abs_goodput_delta_pct", float("nan")))
    rows = int(payload.get("rows", 0) or 0)
    return [{
        "gap": "A5",
        "check": "posalpha_estimator_targeted_audit",
        "max_abs_goodput_delta_pct": round(max_abs, 4),
        "count": rows,
        "pass": int(rows >= 5 and math.isfinite(max_abs) and max_abs <= 2.0),
        "evidence": str(path.relative_to(ROOT)),
        "note": "Limited targeted audit only; Bayes/non-stationary estimator audit remains future work.",
    }]


def markdown(rows: list[dict]) -> str:
    lines = [
        "# TPDS Gap Audit",
        "",
        "This report audits saved results against the reviewer-risk items in `improve_cc.md`.",
        "It is diagnostic and does not modify raw measurements.",
        "",
        "## Summary",
        "",
    ]
    failed = [r for r in rows if str(r.get("pass", "")) in ("0", "False", "false")]
    lines.append(f"- Checks: {len(rows)}")
    lines.append(f"- Failed / still open: {len(failed)}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    fields = ["gap", "check", "result_set", "N", "kv_blocks", "candidate_policy", "candidate_lift_vs_lru_pct", "mean_lift_pct", "ci95_low_pct", "ci95_high_pct", "wins", "total", "observed_Ns", "observed_arrival_models", "observed_model_pairs", "observed_markers", "observed_groups", "safe_min_goodput_ratio_vs_nospec", "unsafe_max_goodput", "nospec_goodput", "specsched_goodput", "safe_goodput", "safe_lift_vs_nospec_pct", "mae_ms", "r2", "max_degradation_pct", "max_abs_goodput_delta_pct", "pass", "note"]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        vals = [str(row.get(field, "")) for field in fields]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    rows: list[dict] = []
    rows.extend(audit_bestfixed(
        ROOT / "results/tpds_multiseed_main/processed/main_multiseed_paired_lifts.csv",
        "closed_loop_main",
    ))
    heterog_path = ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_paired_lifts.csv"
    heterog_summary_path = ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_summary.csv"
    rows.extend(audit_multiseed_completion(heterog_summary_path, "heterog_poisson", {16, 32, 64}, 3))
    if heterog_path.exists():
        rows.extend(audit_bestfixed(heterog_path, "heterog_poisson"))
    rows.extend(audit_poisson_pilot(
        ROOT / "results/heterog_poisson_pilot/processed/scheduler_main_summary.csv",
        heterog_path,
    ))
    rows.extend(audit_system_breadth([
        ROOT / "results/local_v3_kcap3/processed/scheduler_all_summary.csv",
        ROOT / "results/specserve_targeted/processed/scheduler_all_summary.csv",
        ROOT / "results/second_model_targeted/processed/scheduler_all_summary.csv",
        ROOT / "results/tpds_multiseed_main/processed/main_multiseed_summary.csv",
        ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_summary.csv",
    ]))
    rows.extend(audit_edge_practical(
        ROOT / "results/tpds_multiseed_edge_practical/processed/main_multiseed_summary.csv"
    ))
    rows.extend(audit_detector_safety({
        "qwen15b": ROOT / "results/detector_safety/qwen15b/processed/main_multiseed_summary.csv",
        "smol": ROOT / "results/detector_safety/smol/processed/main_multiseed_summary.csv",
        "pythia": ROOT / "results/detector_safety/pythia/processed/main_multiseed_summary.csv",
    }))
    rows.extend(audit_model_pair_pressure(
        {
            "qwen15b": ROOT / "results/tpds_multiseed_second_model_kv_pressure/processed/main_multiseed_summary.csv",
            "pythia": ROOT / "results/tpds_multiseed_pythia_model_kv_pressure/processed/main_multiseed_summary.csv",
        },
        "model_pair_kv_pressure_complete",
        "configured_kv_blocks",
        {"256", "512"},
    ))
    rows.extend(audit_model_pair_pressure(
        {
            "qwen15b": ROOT / "results/tpds_multiseed_second_model_slo_sensitivity/processed/main_multiseed_summary.csv",
            "pythia": ROOT / "results/tpds_multiseed_pythia_model_slo_sensitivity/processed/main_multiseed_summary.csv",
        },
        "model_pair_slo_sensitivity_complete",
        "configured_beta",
        {"1.25", "2.0"},
    ))
    rows.extend(audit_specserve(ROOT / "results/specserve_targeted/processed/scheduler_all_summary.csv"))
    rows.extend(audit_evictv2(
        ROOT / "results/local_v3_kcap3/processed/scheduler_kv_sweep_summary.csv",
        [
            ROOT / "results/evictv2_targeted/processed/scheduler_kv_sweep_summary.csv",
            ROOT / "results/evict_guarded_targeted/processed/scheduler_kv_sweep_summary.csv",
            ROOT / "results/evictv2_extended/processed/scheduler_kv_sweep_summary.csv",
        ],
        [
            ROOT / "results/evict_calibrated_targeted/processed/scheduler_kv_sweep_summary.csv",
            ROOT / "results/evict_calibrated_guardrail_probe/processed/scheduler_kv_sweep_summary.csv",
        ],
    ))
    rows.extend(audit_workload_features(
        [
            ROOT / "results/local_v3_kcap3/processed/scheduler_all_summary.csv",
            ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_summary.csv",
        ],
        ROOT / "results/smoke_poisson/processed/scheduler_all_summary.csv",
    ))
    rows.extend(audit_verify_latency_fit(
        ROOT / "results/verify_latency_fit/processed/verify_latency_fit_metrics.json"
    ))
    rows.extend(audit_estimator_posalpha(
        ROOT / "results/estimator_audit/processed/posalpha_estimator_audit.json"
    ))
    payload = {
        "status": "pass" if all(bool(int(row.get("pass", 0))) for row in rows) else "open_gaps",
        "checks": len(rows),
        "open_gaps": sum(1 for row in rows if not bool(int(row.get("pass", 0)))),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "tpds_gap_audit.csv", rows)
    (out_dir / "tpds_gap_audit.json").write_text(json.dumps({"summary": payload, "checks": rows}, indent=2), encoding="utf-8")
    (out_dir / "tpds_gap_audit.md").write_text(markdown(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(out_dir / "tpds_gap_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

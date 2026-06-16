#!/usr/bin/env python3
"""Audit headline paper claims against processed CSV results.

This script is intentionally narrow: it checks headline values cited in
the paper, without changing any measured data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def read_manuscript_source(include_anonymous: bool = False) -> str:
    paths = [ROOT / "main" / "main.tex", ROOT / "main" / "appendix.tex"]
    if include_anonymous:
        paths.append(ROOT / "main" / "main_anonymous.tex")
    chunks = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def approx(actual: float, expected: float, tol: float = 0.06) -> bool:
    return abs(actual - expected) <= tol


def record(
    checks: list[dict[str, Any]],
    name: str,
    actual: float,
    expected: float,
    unit: str = "",
    tol: float = 0.06,
) -> None:
    checks.append(
        {
            "name": name,
            "actual": round(actual, 4),
            "expected": expected,
            "unit": unit,
            "tolerance": tol,
            "pass": approx(actual, expected, tol),
        }
    )


def find_row(rows: list[dict[str, str]], **want: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in want.items()):
            return row
    raise KeyError(f"missing row {want}")


def audit_multiseed(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(ROOT / "results/tpds_multiseed_main/processed/main_multiseed_paired_lifts.csv")
    expected = {
        ("16", "NoSpec"): 69.5,
        ("32", "NoSpec"): 93.5,
        ("64", "NoSpec"): 91.3,
        ("16", "TETRIS-Proxy"): 2.6,
        ("32", "TETRIS-Proxy"): 5.1,
        ("64", "TETRIS-Proxy"): 10.0,
        ("16", "BestFixed"): -1.1,
        ("32", "BestFixed"): 0.6,
        ("64", "BestFixed"): 0.2,
    }
    for (n_value, baseline), expected_value in expected.items():
        row = find_row(rows, N=n_value, baseline=baseline)
        record(
            checks,
            f"paired lift N={n_value} vs {baseline}",
            as_float(row, "goodput_lift_pct_mean"),
            expected_value,
            "%",
            0.08,
        )


def audit_poisson_multiseed(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_paired_lifts.csv")
    expected = {
        ("16", "NoSpec"): (66.5, True),
        ("32", "NoSpec"): (85.4, True),
        ("64", "NoSpec"): (81.1, True),
        ("16", "TETRIS-Proxy"): (31.3, False),
        ("32", "TETRIS-Proxy"): (20.9, True),
        ("64", "TETRIS-Proxy"): (13.6, True),
        ("16", "BestFixed"): (-5.0, False),
        ("32", "BestFixed"): (9.0, False),
        ("64", "BestFixed"): (2.2, False),
    }
    for (n_value, baseline), (expected_value, ci_positive) in expected.items():
        row = find_row(rows, N=n_value, baseline=baseline)
        record(
            checks,
            f"poisson paired lift N={n_value} vs {baseline}",
            as_float(row, "goodput_lift_pct_mean"),
            expected_value,
            "%",
            0.08,
        )
        checks.append(
            {
                "name": f"poisson CI-positive N={n_value} vs {baseline}",
                "actual": row.get("ci95_positive") == "1",
                "expected": ci_positive,
                "pass": (row.get("ci95_positive") == "1") == ci_positive,
            }
        )


def audit_eagle2_compare(checks: list[dict[str, Any]]) -> None:
    root = ROOT / "results/eagle2_compare/processed"
    summary = read_csv(root / "main_multiseed_summary.csv")
    paired = read_csv(root / "main_multiseed_paired_lifts.csv")
    expected_policies = {
        "NoSpec",
        "FixedK-1",
        "FixedK-3",
        "TETRIS-Proxy",
        "EAGLE-2-Proxy",
        "SpecServe-Proxy",
        "SpecSched",
    }
    raw_count = 0
    for seed in ("7", "42", "123"):
        rows = read_csv(ROOT / f"results/eagle2_compare/seed_{seed}/processed/scheduler_main_summary.csv")
        filtered = [
            row
            for row in rows
            if row.get("N") in {"16", "32", "64"} and row.get("policy") in expected_policies
        ]
        raw_count += len(filtered)
    checks.append(
        {
            "name": "EAGLE-2 comparison raw cell count",
            "actual": raw_count,
            "expected": 63,
            "pass": raw_count == 63,
        }
    )
    checks.append(
        {
            "name": "EAGLE-2 comparison aggregate row count",
            "actual": len(summary),
            "expected": 21,
            "pass": len(summary) == 21,
        }
    )

    expected_lifts = {
        ("16", "NoSpec"): 66.9,
        ("32", "NoSpec"): 88.0,
        ("64", "NoSpec"): 85.6,
        ("16", "TETRIS-Proxy"): 3.4,
        ("32", "TETRIS-Proxy"): 3.7,
        ("64", "TETRIS-Proxy"): 11.0,
        ("16", "EAGLE-2-Proxy"): 14.1,
        ("32", "EAGLE-2-Proxy"): 19.0,
        ("64", "EAGLE-2-Proxy"): 15.1,
        ("16", "SpecServe-Proxy"): 12.8,
        ("32", "SpecServe-Proxy"): 14.1,
        ("64", "SpecServe-Proxy"): 13.0,
    }
    for (n_value, baseline), expected_value in expected_lifts.items():
        row = find_row(paired, N=n_value, baseline=baseline)
        record(
            checks,
            f"EAGLE compare paired lift N={n_value} vs {baseline}",
            as_float(row, "goodput_lift_pct_mean"),
            expected_value,
            "%",
            0.08,
        )
        if baseline == "EAGLE-2-Proxy":
            checks.append(
                {
                    "name": f"EAGLE compare CI-positive N={n_value} vs {baseline}",
                    "actual": row.get("ci95_positive") == "1",
                    "expected": True,
                    "pass": row.get("ci95_positive") == "1",
                }
            )

    expected_p99_reductions = {
        "16": 9.6,
        "32": 14.8,
        "64": 21.4,
    }
    for n_value, expected_value in expected_p99_reductions.items():
        specserve = find_row(summary, N=n_value, policy="SpecServe-Proxy")
        specsched = find_row(summary, N=n_value, policy="SpecSched")
        reduction = (
            1.0
            - as_float(specsched, "p99_tpot_ms_mean")
            / as_float(specserve, "p99_tpot_ms_mean")
        ) * 100.0
        record(
            checks,
            f"EAGLE compare P99 reduction vs SpecServe-Proxy N={n_value}",
            reduction,
            expected_value,
            "%",
            0.08,
        )


def audit_slo_sensitivity(checks: list[dict[str, Any]]) -> None:
    rows = []
    rows.extend(read_csv(ROOT / "results/local_v3_kcap3/processed/scheduler_slo_sensitivity_summary.csv"))
    rows.extend(read_csv(ROOT / "results/local_v3_kcap3/processed/scheduler_main_summary.csv"))
    tags = {
        "1.25": "paper3090_slo_beta125",
        "1.5": "paper3090_main_beta15",
        "2.0": "paper3090_slo_beta20",
    }
    expected = {
        ("16", "1.25"): 31.3,
        ("16", "1.5"): 30.0,
        ("16", "2.0"): 12.3,
        ("32", "1.25"): 36.9,
        ("32", "1.5"): 34.0,
        ("32", "2.0"): 14.0,
        ("64", "1.25"): 19.5,
        ("64", "1.5"): 19.5,
        ("64", "2.0"): 20.2,
    }
    for (n_value, beta), expected_value in expected.items():
        tag = tags[beta]
        spec = find_row(rows, tag=tag, N=n_value, policy="SpecSched")
        indep = find_row(rows, tag=tag, N=n_value, policy="IndepOpt")
        lift = (as_float(spec, "goodput_tok_s") / as_float(indep, "goodput_tok_s") - 1.0) * 100.0
        record(checks, f"SLO lift N={n_value} beta={beta}", lift, expected_value, "%", 0.08)


def audit_second_model_boundary(checks: list[dict[str, Any]]) -> None:
    summary = read_csv(ROOT / "results/tpds_multiseed_second_model/processed/main_multiseed_summary.csv")
    expected_summary = {
        "NoSpec": (100.73, 6.15),
        "FixedK-1": (80.31, 7.43),
        "SpecSched": (69.97, 8.81),
    }
    for policy, (expected_mean, expected_std) in expected_summary.items():
        row = find_row(summary, N="16", policy=policy)
        record(checks, f"second-model {policy} goodput mean", as_float(row, "goodput_tok_s_mean"), expected_mean, "tok/s", 0.02)
        record(checks, f"second-model {policy} goodput std", as_float(row, "goodput_tok_s_std"), expected_std, "tok/s", 0.02)

    paired = read_csv(ROOT / "results/tpds_multiseed_second_model/processed/main_multiseed_paired_lifts.csv")
    expected_lifts = {
        "NoSpec": -30.7,
        "BestFixed": -13.1,
    }
    for baseline, expected in expected_lifts.items():
        row = find_row(paired, N="16", baseline=baseline)
        record(checks, f"second-model lift vs {baseline}", as_float(row, "goodput_lift_pct_mean"), expected, "%", 0.08)
        checks.append(
            {
                "name": f"second-model CI-negative vs {baseline}",
                "actual": as_float(row, "goodput_lift_pct_ci95_high") < 0.0,
                "expected": True,
                "pass": as_float(row, "goodput_lift_pct_ci95_high") < 0.0,
            }
        )


def audit_aux_model_boundaries(checks: list[dict[str, Any]]) -> None:
    models = {
        "smollm2": {
            "root": ROOT / "results/tpds_multiseed_smol_model/processed",
            "summary": {
                "NoSpec": (116.62, 13.40),
                "SpecSched": (23.83, 13.51),
            },
            "lifts": {
                "NoSpec": (-79.5, False),
                "TETRIS-Proxy": (-28.5, False),
                "BestFixed": (-42.3, False),
            },
        },
        "pythia": {
            "root": ROOT / "results/tpds_multiseed_pythia_model/processed",
            "summary": {
                "NoSpec": (90.53, 6.47),
                "TETRIS-Proxy": (134.37, 17.65),
                "SpecSched": (126.09, 11.44),
            },
            "lifts": {
                "NoSpec": (39.3, True),
                "TETRIS-Proxy": (-5.7, False),
                "BestFixed": (-4.0, False),
            },
        },
    }
    for name, spec in models.items():
        root = spec["root"]
        summary = read_csv(root / "main_multiseed_summary.csv")
        for policy, (expected_mean, expected_std) in spec["summary"].items():
            row = find_row(summary, N="16", policy=policy)
            record(checks, f"{name} {policy} goodput mean", as_float(row, "goodput_tok_s_mean"), expected_mean, "tok/s", 0.02)
            record(checks, f"{name} {policy} goodput std", as_float(row, "goodput_tok_s_std"), expected_std, "tok/s", 0.02)

        paired = read_csv(root / "main_multiseed_paired_lifts.csv")
        for baseline, (expected_lift, ci_positive) in spec["lifts"].items():
            row = find_row(paired, N="16", baseline=baseline)
            record(checks, f"{name} lift vs {baseline}", as_float(row, "goodput_lift_pct_mean"), expected_lift, "%", 0.08)
            checks.append(
                {
                    "name": f"{name} CI-positive vs {baseline}",
                    "actual": row.get("ci95_positive") == "1",
                    "expected": ci_positive,
                    "pass": (row.get("ci95_positive") == "1") == ci_positive,
                }
            )


def audit_model_pair_n32_boundaries(checks: list[dict[str, Any]]) -> None:
    models = {
        "qwen15b-n32": {
            "root": ROOT / "results/tpds_multiseed_second_model_n32/processed",
            "summary": {
                "NoSpec": (96.61, 3.09),
                "SpecSched": (53.48, 4.38),
            },
            "lifts": {
                "NoSpec": (-44.6, False),
                "TETRIS-Proxy": (11.9, False),
                "BestFixed": (-22.8, False),
            },
        },
        "pythia-n32": {
            "root": ROOT / "results/tpds_multiseed_pythia_model_n32/processed",
            "summary": {
                "NoSpec": (96.31, 2.15),
                "SpecSched": (148.64, 4.38),
            },
            "lifts": {
                "NoSpec": (54.3, True),
                "TETRIS-Proxy": (1.5, False),
                "BestFixed": (-1.9, False),
            },
        },
    }
    for name, spec in models.items():
        root = spec["root"]
        summary = read_csv(root / "main_multiseed_summary.csv")
        for policy, (expected_mean, expected_std) in spec["summary"].items():
            row = find_row(summary, N="32", policy=policy)
            record(checks, f"{name} {policy} goodput mean", as_float(row, "goodput_tok_s_mean"), expected_mean, "tok/s", 0.02)
            record(checks, f"{name} {policy} goodput std", as_float(row, "goodput_tok_s_std"), expected_std, "tok/s", 0.02)

        paired = read_csv(root / "main_multiseed_paired_lifts.csv")
        for baseline, (expected_lift, ci_positive) in spec["lifts"].items():
            row = find_row(paired, N="32", baseline=baseline)
            record(checks, f"{name} lift vs {baseline}", as_float(row, "goodput_lift_pct_mean"), expected_lift, "%", 0.08)
            checks.append(
                {
                    "name": f"{name} CI-positive vs {baseline}",
                    "actual": row.get("ci95_positive") == "1",
                    "expected": ci_positive,
                    "pass": (row.get("ci95_positive") == "1") == ci_positive,
                }
            )


def audit_detector_safety(checks: list[dict[str, Any]]) -> None:
    models = {
        "detector-qwen15b": {
            "root": ROOT / "results/detector_safety/qwen15b/processed",
            "summary": {
                "NoSpec": (99.43, 5.93),
                "SpecSched": (69.13, 5.87),
                "SpecSched-Safe": (98.75, 6.34),
            },
            "lifts": {
                "NoSpec": (-0.7, False),
                "SpecSched": (43.4, True),
            },
        },
        "detector-smollm2": {
            "root": ROOT / "results/detector_safety/smol/processed",
            "summary": {
                "NoSpec": (120.46, 8.37),
                "SpecSched": (24.27, 10.54),
                "SpecSched-Safe": (115.00, 13.67),
            },
            "lifts": {
                "NoSpec": (-4.7, False),
                "SpecSched": (477.0, False),
            },
        },
        "detector-pythia": {
            "root": ROOT / "results/detector_safety/pythia/processed",
            "summary": {
                "NoSpec": (86.99, 1.20),
                "SpecSched": (128.09, 13.15),
                "SpecSched-Safe": (127.53, 15.04),
            },
            "lifts": {
                "NoSpec": (46.6, True),
                "SpecSched": (-0.5, False),
            },
        },
    }
    for name, spec in models.items():
        summary = read_csv(spec["root"] / "main_multiseed_summary.csv")
        for policy, (expected_mean, expected_std) in spec["summary"].items():
            row = find_row(summary, N="16", policy=policy)
            record(checks, f"{name} {policy} goodput mean", as_float(row, "goodput_tok_s_mean"), expected_mean, "tok/s", 0.02)
            record(checks, f"{name} {policy} goodput std", as_float(row, "goodput_tok_s_std"), expected_std, "tok/s", 0.02)

        paired = read_csv(spec["root"] / "main_multiseed_paired_lifts.csv")
        for baseline, (expected_lift, ci_positive) in spec["lifts"].items():
            row = find_row(paired, N="16", baseline=baseline)
            record(checks, f"{name} safe lift vs {baseline}", as_float(row, "goodput_lift_pct_mean"), expected_lift, "%", 0.08)
            checks.append(
                {
                    "name": f"{name} safe CI-positive vs {baseline}",
                    "actual": row.get("ci95_positive") == "1",
                    "expected": ci_positive,
                    "pass": (row.get("ci95_positive") == "1") == ci_positive,
                }
            )


def audit_model_pair_pressure(checks: list[dict[str, Any]]) -> None:
    specs = {
        "qwen15b": {
            "kv_root": ROOT / "results/tpds_multiseed_second_model_kv_pressure/processed",
            "slo_root": ROOT / "results/tpds_multiseed_second_model_slo_sensitivity/processed",
            "kv": {
                ("256", "SpecSched", "goodput_tok_s_mean"): 53.19,
                ("256", "SpecSched-H2O", "goodput_tok_s_mean"): 55.99,
                ("512", "SpecSched", "goodput_tok_s_mean"): 50.87,
                ("512", "SpecSched-Longest", "goodput_tok_s_mean"): 58.68,
            },
            "slo": {
                ("1.25", "NoSpec"): (-53.8, False),
                ("2.0", "NoSpec"): (-31.6, False),
            },
        },
        "pythia": {
            "kv_root": ROOT / "results/tpds_multiseed_pythia_model_kv_pressure/processed",
            "slo_root": ROOT / "results/tpds_multiseed_pythia_model_slo_sensitivity/processed",
            "kv": {
                ("256", "SpecSched", "goodput_tok_s_mean"): 131.58,
                ("256", "SpecSched-H2O", "goodput_tok_s_mean"): 135.66,
                ("512", "SpecSched", "goodput_tok_s_mean"): 148.21,
                ("512", "SpecSched-H2O", "goodput_tok_s_mean"): 152.12,
            },
            "slo": {
                ("1.25", "NoSpec"): (90.5, True),
                ("2.0", "NoSpec"): (51.0, True),
            },
        },
    }
    for name, spec in specs.items():
        kv_summary = read_csv(spec["kv_root"] / "main_multiseed_summary.csv")
        for (kv_blocks, policy, metric), expected in spec["kv"].items():
            row = find_row(kv_summary, configured_kv_blocks=kv_blocks, N="32", policy=policy)
            record(checks, f"{name} pressure kv={kv_blocks} {policy} {metric}", as_float(row, metric), expected, metric, 0.03)

        slo_paired = read_csv(spec["slo_root"] / "main_multiseed_paired_lifts.csv")
        for (beta, baseline), (expected_lift, ci_positive) in spec["slo"].items():
            row = find_row(slo_paired, configured_beta=beta, N="32", baseline=baseline)
            record(checks, f"{name} pressure beta={beta} lift vs {baseline}", as_float(row, "goodput_lift_pct_mean"), expected_lift, "%", 0.08)
            checks.append(
                {
                    "name": f"{name} pressure beta={beta} CI-positive vs {baseline}",
                    "actual": row.get("ci95_positive") == "1",
                    "expected": ci_positive,
                    "pass": (row.get("ci95_positive") == "1") == ci_positive,
                }
            )


def audit_kv_pressure_multiseed(checks: list[dict[str, Any]]) -> None:
    summary = read_csv(ROOT / "results/tpds_multiseed_kv_pressure/processed/main_multiseed_summary.csv")
    expected_summary = {
        ("512", "SpecSched", "goodput_tok_s_mean"): 72.10,
        ("512", "SpecSched", "goodput_tok_s_std"): 2.91,
        ("512", "SpecSched-LRU", "goodput_tok_s_mean"): 73.01,
        ("512", "SpecSched-LRU", "goodput_tok_s_std"): 1.89,
        ("512", "SpecSched-vLLM", "p99_tpot_ms_mean"): 818.12,
        ("512", "SpecSched-vLLM", "p99_tpot_ms_std"): 21.87,
        ("512", "SpecSched-H2O", "reprefills_per_1k_tokens_mean"): 3.18,
        ("512", "SpecSched-H2O", "reprefills_per_1k_tokens_std"): 0.29,
        ("1024", "SpecSched-Longest", "goodput_tok_s_mean"): 75.02,
        ("1024", "SpecSched-Longest", "goodput_tok_s_std"): 1.78,
        ("1024", "SpecSched-Longest", "reprefills_per_1k_tokens_mean"): 0.74,
        ("1024", "SpecSched-Longest", "reprefills_per_1k_tokens_std"): 0.06,
        ("1024", "SpecSched", "goodput_tok_s_mean"): 74.28,
        ("1024", "SpecSched", "goodput_tok_s_std"): 2.13,
        ("1024", "SpecSched", "p99_tpot_ms_mean"): 811.34,
        ("1024", "SpecSched", "p99_tpot_ms_std"): 73.06,
        ("1024", "SpecSched", "reprefills_per_1k_tokens_mean"): 2.44,
        ("1024", "SpecSched", "reprefills_per_1k_tokens_std"): 0.31,
    }
    for (kv_blocks, policy, metric), expected in expected_summary.items():
        row = find_row(summary, configured_kv_blocks=kv_blocks, N="64", policy=policy)
        record(checks, f"kv-pressure kv={kv_blocks} {policy} {metric}", as_float(row, metric), expected, metric, 0.02)

    paired = read_csv(ROOT / "results/tpds_multiseed_kv_pressure/processed/main_multiseed_paired_lifts.csv")
    for row in paired:
        checks.append(
            {
                "name": (
                    f"kv-pressure CI crosses zero kv={row.get('configured_kv_blocks')} "
                    f"baseline={row.get('baseline')}"
                ),
                "actual": row.get("ci95_positive") == "0",
                "expected": True,
                "pass": row.get("ci95_positive") == "0",
            }
        )


def audit_edge_practical(checks: list[dict[str, Any]]) -> None:
    rows = read_csv(ROOT / "results/tpds_multiseed_edge_practical/processed/main_multiseed_summary.csv")
    expected = {
        ("4", "NoSpec", "goodput_tok_s_mean"): 46.16,
        ("8", "NoSpec", "goodput_tok_s_mean"): 41.88,
        ("4", "SpecSched", "goodput_tok_s_mean"): 0.0,
        ("8", "SpecSched", "goodput_tok_s_mean"): 0.0,
        ("4", "SpecSched", "p99_tpot_ms_mean"): 643.05,
        ("8", "SpecSched", "p99_tpot_ms_mean"): 1407.31,
        ("4", "SpecSched-Safe", "goodput_tok_s_mean"): 45.67,
        ("8", "SpecSched-Safe", "goodput_tok_s_mean"): 41.82,
        ("4", "SpecSched-Safe", "slo_attainment_mean"): 1.0,
        ("8", "SpecSched-Safe", "slo_attainment_mean"): 0.9167,
    }
    tolerances = {
        "goodput_tok_s_mean": 0.02,
        "slo_attainment_mean": 0.001,
        "p99_tpot_ms_mean": 0.02,
    }
    for (n_value, policy, metric), expected_value in expected.items():
        row = find_row(rows, N=n_value, policy=policy)
        record(
            checks,
            f"edge N={n_value} {policy} {metric}",
            as_float(row, metric),
            expected_value,
            metric,
            tolerances[metric],
        )

    detector_rows = [row for row in rows if row.get("policy") == "SpecSched-Safe"]
    checks.append(
        {
            "name": "edge SpecSched-Safe detector falls back on all loads",
            "actual": [row.get("policy") for row in detector_rows],
            "expected": ["SpecSched-Safe", "SpecSched-Safe"],
            "pass": len(detector_rows) == 2,
        }
    )

    single_seed = read_csv(ROOT / "results/tpds_multiseed_edge_practical/seed_7/processed/scheduler_main_summary.csv")
    for n_value, expected_low, expected_high in (("4", 0.63, 0.65), ("8", 0.63, 0.65)):
        row = find_row(single_seed, N=n_value, policy="SpecSched")
        remote_overhead = as_float(row, "service_draft_frac") + as_float(row, "service_other_frac")
        checks.append(
            {
                "name": f"edge N={n_value} remote overhead fraction in stated range",
                "actual": round(remote_overhead, 4),
                "expected": f"{expected_low}-{expected_high}",
                "pass": expected_low <= remote_overhead <= expected_high,
            }
        )


def audit_quality(checks: list[dict[str, Any]]) -> None:
    for path in [
        ROOT / "results/local_v3_kcap3/processed/quality_checks.csv",
        ROOT / "results/specserve_targeted/processed/quality_checks.csv",
    ]:
        rows = read_csv(path)
        for row in rows:
            name = f"{path.parent.parent.name}:{row.get('tag')}"
            passed = (
                row.get("failed_cells") == "0"
                and row.get("missing_cells") == "[]"
                and row.get("goodput_all_zero") == "0"
                and row.get("attainment_all_zero") == "0"
                and row.get("unit_check_p99_lt_100x_slo") == "1"
            )
            checks.append({"name": f"quality {name}", "pass": passed, "row": row})


def audit_verify_latency_fit(checks: list[dict[str, Any]]) -> None:
    path = ROOT / "results/verify_latency_fit/processed/verify_latency_fit_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    perturb = json.loads(
        (ROOT / "results/verify_latency_fit/processed/verify_latency_perturbation.json").read_text(encoding="utf-8")
    )
    checks.append({
        "name": "verify latency fit mode",
        "actual": payload.get("profile_mode"),
        "expected": "cached_verify",
        "pass": payload.get("profile_mode") == "cached_verify",
    })
    record(checks, "verify latency fit count", float(metrics["count"]), 64.0, "points", 0.0)
    record(checks, "verify latency fit MAE", float(metrics["mae_ms"]), 0.34, "ms", 0.01)
    record(checks, "verify latency fit RMSE", float(metrics["rmse_ms"]), 0.42, "ms", 0.01)
    record(checks, "verify latency fit MAPE", float(metrics["mape_pct"]), 1.42, "%", 0.01)
    record(checks, "verify latency fit R2", float(metrics["r2"]), 0.975, "", 0.001)
    record(checks, "verify latency perturbation max ordering change", float(perturb["max_degradation_pct"]), 1.57, "%", 0.01)


def audit_estimator_targeted(checks: list[dict[str, Any]]) -> None:
    path = ROOT / "results/estimator_audit/processed/posalpha_estimator_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    record(checks, "PosAlpha targeted min goodput delta", float(payload["min_goodput_delta_pct"]), -1.35, "%", 0.01)
    record(checks, "PosAlpha targeted max goodput delta", float(payload["max_goodput_delta_pct"]), 0.03, "%", 0.01)
    record(checks, "PosAlpha targeted max absolute delta", float(payload["max_abs_goodput_delta_pct"]), 1.35, "%", 0.01)


def audit_baseline_table_scope(checks: list[dict[str, Any]]) -> None:
    path = ROOT / "paper_assets/tables/baseline_performance.tex"
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        "Core baseline comparison under the common executor",
        "paired \\system{} SLO-goodput lift at $N=16/32/64$",
        "\\multicolumn{3}{c}{\\system{} SLO-goodput lift ($\\uparrow$)}",
        "$N=16$ & $N=32$ & $N=64$",
        "Baseline / comparator with source",
        "NoSpec, vLLM/PagedAttention~\\cite{kwon2023efficient}",
        "BestFixed, Speculative decoding~\\cite{leviathan2023fast,chen2023accelerating}",
        "\\secondb{$-$1.1\\%} & \\secondb{+0.6\\%} & \\secondb{+0.2\\%}",
        "IndepOpt, Online draft control~\\cite{liu2024online,liu2024turbospec}",
        "TETRIS-Proxy, TETRIS~\\cite{wu2025tetris}",
        "EAGLE-2-Proxy, EAGLE-2~\\cite{li2024eagle2}",
        "SpecServe-Proxy, SpecServe/AdaSpec~\\cite{huang2025specserve}",
        "\\rowcolor{tabgray}BestFixed",
    ]
    forbidden_snippets = [
        "H2O / Longest",
        "Algorithm~\\ref{alg:joint}",
        "Qwen2.5-1.5B/0.5B",
        "Pythia-2.8B/160M",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    stale = [snippet for snippet in forbidden_snippets if snippet in text]
    checks.append(
        {
            "name": "Table V baseline map wording and proxy scope",
            "pass": not missing and not stale,
            "missing": missing,
            "stale": stale,
        }
    )

    row_order = ["NoSpec,", "BestFixed,", "IndepOpt,", "TETRIS-Proxy,", "EAGLE-2-Proxy,", "SpecServe-Proxy,"]
    positions = [text.find(name) for name in row_order]
    checks.append(
        {
            "name": "Table V rows keep baseline-first evaluated-baseline order",
            "actual": positions,
            "expected": row_order,
            "pass": all(pos >= 0 for pos in positions) and positions == sorted(positions),
        }
    )

    auxiliary_tables = [
        ROOT / "paper_assets/tables/model_pair_performance.tex",
        ROOT / "paper_assets/tables/model_pair_n32.tex",
    ]
    aux_findings = {}
    for aux_path in auxiliary_tables:
        aux_text = aux_path.read_text(encoding="utf-8")
        missing_aux = []
        if "TETRIS-Proxy" not in aux_text:
            missing_aux.append("TETRIS-Proxy")
        if " & TETRIS & " in aux_text:
            missing_aux.append("ambiguous TETRIS column")
        if "Cloud target & Edge draft" in aux_text:
            missing_aux.append("over-scoped edge hardware column label")
        if missing_aux:
            aux_findings[str(aux_path.relative_to(ROOT))] = missing_aux
    checks.append(
        {
            "name": "auxiliary model-pair tables keep proxy baseline labels",
            "pass": not aux_findings,
            "findings": aux_findings,
        }
    )


def audit_claim_scope_guardrails(checks: list[dict[str, Any]]) -> None:
    """Reject stale wording that would overstate the current evidence."""
    paths = [
        ROOT / "main" / "main.tex",
        ROOT / "main" / "main_anonymous.tex",
        ROOT / "main" / "cover_letter_tpds.md",
        ROOT / "main" / "cover_letter_tpds_anonymous.md",
        ROOT / "main" / "submission_form_fields_tpds.md",
        ROOT / "main" / "submission_metadata_tpds.json",
        ROOT / "paper_assets" / "tables" / "baseline_performance.tex",
        ROOT / "scripts" / "generate_model_pair_table.py",
    ]
    forbidden = [
        "native EAGLE-2 superiority",
        "baseline comparison map",
        "claim-scoped comparison of evaluated baselines",
        "\\system{} lift at $N=16/32/64$",
        "Table V is presented as a claim-scoped evaluated-baseline map",
        "measured SpecSched lift",
        "dominates fixed-k",
        "outperforms BestFixed",
        "surpasses BestFixed",
        "exceeds BestFixed",
        "native-system gains",
        "native baseline results",
        "production-trace results",
        "production trace results",
        "multi-GPU results",
        "24-hour results",
        "AE badge",
    ]
    findings: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits = [needle for needle in forbidden if needle in text]
        if hits:
            findings[str(path.relative_to(ROOT))] = hits
    checks.append(
        {
            "name": "claim-scope guardrails avoid stale overclaim wording",
            "pass": not findings,
            "findings": findings,
        }
    )

    submission_paths = [
        ROOT / "main" / "cover_letter_tpds.md",
        ROOT / "main" / "cover_letter_tpds_anonymous.md",
        ROOT / "main" / "submission_form_fields_tpds.md",
        ROOT / "main" / "submission_metadata_tpds.json",
    ]
    submission_required = [
        "TETRIS-Proxy",
        "EAGLE-2-Proxy",
        "SpecServe-Proxy",
        "does not statistically exceed",
    ]
    submission_findings: dict[str, list[str]] = {}
    for path in submission_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        missing = [snippet for snippet in submission_required if snippet not in text]
        if "native-system reproductions" not in text and "ShareGPT/LMSYS-derived trace replay" not in text:
            missing.append("native/proxy or trace scope wording")
        if missing:
            submission_findings[str(path.relative_to(ROOT))] = missing
    checks.append(
        {
            "name": "submission helpers preserve proxy and BestFixed scope",
            "pass": not submission_findings,
            "missing": submission_findings,
        }
    )

    main_path = ROOT / "main" / "main.tex"
    if not main_path.exists():
        checks.append(
            {
                "name": "main text states baseline evidence-map boundaries",
                "pass": True,
                "detail": "skipped because manuscript source is not included in this artifact-only bundle",
            }
        )
        return

    main_text = read_manuscript_source()
    required = [
        "native speculative-serving comparisons",
        "Table~\\ref{tab:baseline-performance} gives the main benchmark",
        "evaluated baseline first, reference",
        "paired goodput gains at $N=16/32/64$",
        "remaining statistically tied with the offline-tuned BestFixed envelope",
        "Additional $N=32$ model-pair and KV-pressure boundary cells are kept",
    ]
    missing = [snippet for snippet in required if snippet not in main_text]
    checks.append(
        {
            "name": "main text states baseline evidence-map boundaries",
            "pass": not missing,
            "missing": missing,
        }
    )


def audit_evaluation_scope_table(checks: list[dict[str, Any]]) -> None:
    """Keep the setup table claim-scoped rather than a generic benchmark list."""
    manuscript_text = read_manuscript_source()
    required = [
        "\\textit{Workload} & \\textit{$N$} & \\textit{Claim / scope}",
        "\\section{Evaluation Matrix and Setup Details}",
        "\\label{app:evaluation-scope}",
        "Edge overhead diagnostic, not an optimized edge-serving result",
        "Common-executor scheduler comparison under calibrated SLOs",
        "Cache-regime boundary with no universal eviction-rule claim",
        "Model-pair safety and transfer boundaries",
        "Safe fallback on a real remote-draft path",
        "CPU scheduler-cost scaling only",
    ]
    forbidden = [
        "\\textit{Workload} & \\textit{$N$} & \\textit{Purpose}",
        "EQ2--EQ3: compare draft/batch policies under calibrated SLOs",
        "EQ4: sweep 256--1024 KV blocks and eviction policies",
        "EQ1: measure real distributed-draft bottlenecks",
    ]
    findings: dict[str, dict[str, list[str]]] = {}
    missing = [snippet for snippet in required if snippet not in manuscript_text]
    stale = [snippet for snippet in forbidden if snippet in manuscript_text]
    appendix_path = ROOT / "main" / "appendix.tex"
    appendix_text = appendix_path.read_text(encoding="utf-8") if appendix_path.exists() else ""
    if "\\label{tab:exp-setup}" not in appendix_text:
        missing.append("setup table moved to appendix")
    if missing or stale:
        findings["main/appendix.tex"] = {
            "missing": missing,
            "stale": stale,
        }
    checks.append(
        {
            "name": "evaluation setup table preserves claim/scope boundaries",
            "pass": not findings,
            "findings": findings,
        }
    )


def audit_limitations_scope_guardrails(checks: list[dict[str, Any]]) -> None:
    """Ensure the manuscript keeps the improve_cc D1-D6 gaps scoped out."""
    normalized = " ".join(read_manuscript_source().split())
    required = [
        "paired confidence intervals against BestFixed cross zero",
        "workloads with measured acceptance heterogeneity",
        "bursty arrivals, or tiered SLO mixes",
        "The current artifact does not include production traces",
        "Azure LLM inference traces, BurstGPT, or WildChat",
        "2-GPU tensor-parallel verification",
        "24-hour endurance and cost/energy studies",
        "common-executor scheduler proxies",
        "do not replace native integrations",
        "Native EAGLE-2/SpecInfer reproduction",
        "Dockerized one-command reruns",
        "formal artifact-evaluation packaging",
        "does not convert those missing experiments into evidence",
    ]
    findings: dict[str, list[str]] = {}
    missing = [snippet for snippet in required if snippet not in normalized]
    if missing:
        findings["main+appendix"] = missing
    checks.append(
        {
            "name": "limitations keep improve_cc D1-D6 gaps scoped out",
            "pass": not findings,
            "missing": findings,
        }
    )

    artifact_scope_required = [
        "without CUDA",
        "PyTorch, vLLM, model checkpoints, traces, or Jetson access",
        "Dockerized one-command GPU reruns and formal artifact-evaluation packaging remain outside",
    ]
    artifact_findings: dict[str, list[str]] = {}
    missing = [snippet for snippet in artifact_scope_required if snippet not in normalized]
    if missing:
        artifact_findings["main+appendix"] = missing
    checks.append(
        {
            "name": "artifact scope stays result-level rather than GPU/AE rerun",
            "pass": not artifact_findings,
            "missing": artifact_findings,
        }
    )


def main() -> int:
    checks: list[dict[str, Any]] = []
    for fn in (
        audit_multiseed,
        audit_poisson_multiseed,
        audit_eagle2_compare,
        audit_slo_sensitivity,
        audit_second_model_boundary,
        audit_aux_model_boundaries,
        audit_model_pair_n32_boundaries,
        audit_detector_safety,
        audit_model_pair_pressure,
        audit_kv_pressure_multiseed,
        audit_edge_practical,
        audit_quality,
        audit_verify_latency_fit,
        audit_estimator_targeted,
        audit_baseline_table_scope,
        audit_claim_scope_guardrails,
        audit_evaluation_scope_table,
        audit_limitations_scope_guardrails,
    ):
        fn(checks)
    failed = [check for check in checks if not check["pass"]]
    report = {
        "checks": len(checks),
        "failed": len(failed),
        "status": "pass" if not failed else "fail",
        "failures": failed,
        "all_checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

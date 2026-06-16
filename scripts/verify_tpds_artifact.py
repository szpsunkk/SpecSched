#!/usr/bin/env python3
"""Verify result-level artifact completeness for the TPDS submission.

The check is read-only. It does not rerun GPU experiments and does not
rewrite figures or tables.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "main"
PAPER_ASSETS = ROOT / "paper_assets"
MANIFEST = PAPER_ASSETS / "tables" / "plot_manifest.json"

REQUIRED_CONFIGS = [
    ROOT / "configs" / "paper3090_v3_kcap3.json",
    ROOT / "configs" / "tpds_multiseed_main.json",
    ROOT / "configs" / "tpds_multiseed_eagle2_compare.json",
    ROOT / "configs" / "paper3090_heterog_poisson.json",
    ROOT / "configs" / "tpds_multiseed_heterog_poisson.json",
    ROOT / "configs" / "paper3090_specserve_targeted.json",
    ROOT / "configs" / "paper3090_evictv2_targeted.json",
    ROOT / "configs" / "paper3090_evict_guarded_targeted.json",
    ROOT / "configs" / "paper3090_second_model_targeted.json",
    ROOT / "configs" / "paper3090_second_model_n32.json",
    ROOT / "configs" / "paper3090_pythia_model_n32.json",
    ROOT / "configs" / "tpds_multiseed_second_model_n32.json",
    ROOT / "configs" / "tpds_multiseed_pythia_model_n32.json",
    ROOT / "configs" / "paper3090_second_model_kv_pressure.json",
    ROOT / "configs" / "paper3090_second_model_slo_sensitivity.json",
    ROOT / "configs" / "paper3090_pythia_model_kv_pressure.json",
    ROOT / "configs" / "paper3090_pythia_model_slo_sensitivity.json",
    ROOT / "configs" / "tpds_multiseed_second_model_kv_pressure.json",
    ROOT / "configs" / "tpds_multiseed_second_model_slo_sensitivity.json",
    ROOT / "configs" / "tpds_multiseed_pythia_model_kv_pressure.json",
    ROOT / "configs" / "tpds_multiseed_pythia_model_slo_sensitivity.json",
    ROOT / "configs" / "paper3090_edge_practical.json",
    ROOT / "configs" / "tpds_multiseed_edge_practical.json",
    ROOT / "configs" / "scheduler_overhead_micro.json",
    ROOT / "configs" / "route_kv_stress_sim.json",
    ROOT / "configs" / "bestfixed_stress_sim.json",
    ROOT / "configs" / "paper3090_transfer_gpu.json",
    ROOT / "configs" / "smoke_transfer_gpu.json",
    ROOT / "configs" / "smoke_poisson_openloop_paper.json",
    ROOT / "configs" / "smoke_trace_replay_paper.json",
    ROOT / "data" / "synthetic_production_trace_smoke.jsonl",
]

REQUIRED_PROCESSED = [
    ROOT / "results" / "local_v3_kcap3" / "processed" / "scheduler_all_summary.csv",
    ROOT / "results" / "local_v3_kcap3" / "processed" / "scheduler_main_summary.csv",
    ROOT / "results" / "local_v3_kcap3" / "processed" / "scheduler_kv_sweep_summary.csv",
    ROOT / "results" / "local_v3_kcap3" / "processed" / "scheduler_slo_sensitivity_summary.csv",
    ROOT / "results" / "local_v3_kcap3" / "processed" / "ablation_summary.csv",
    ROOT / "results" / "local_v3_kcap3" / "processed" / "quality_checks.csv",
    ROOT / "results" / "specserve_targeted" / "processed" / "scheduler_all_summary.csv",
    ROOT / "results" / "specserve_targeted" / "processed" / "quality_checks.csv",
    ROOT / "results" / "tpds_multiseed_main" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_main" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "eagle2_compare" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "eagle2_compare" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_kv_pressure" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_kv_pressure" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "evictv2_targeted" / "processed" / "scheduler_kv_sweep_summary.csv",
    ROOT / "results" / "evict_guarded_targeted" / "processed" / "scheduler_kv_sweep_summary.csv",
    ROOT / "results" / "second_model_targeted" / "processed" / "scheduler_all_summary.csv",
    ROOT / "results" / "tpds_multiseed_second_model" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_second_model" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_smol_model" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_smol_model" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_second_model_kv_pressure" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_second_model_kv_pressure" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_second_model_slo_sensitivity" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_second_model_slo_sensitivity" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_kv_pressure" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_kv_pressure" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_slo_sensitivity" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_slo_sensitivity" / "processed" / "main_multiseed_paired_lifts.csv",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "detector_safety" / "qwen15b" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "detector_safety" / "smol" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "detector_safety" / "pythia" / "processed" / "main_multiseed_summary.csv",
    ROOT / "results" / "verify_latency_fit" / "processed" / "verify_latency_fit_metrics.json",
    ROOT / "results" / "verify_latency_fit" / "processed" / "verify_latency_perturbation.json",
    ROOT / "results" / "verify_latency_fit" / "processed" / "verify_latency_fit.csv",
    ROOT / "results" / "estimator_audit" / "processed" / "posalpha_estimator_audit.json",
    ROOT / "results" / "estimator_audit" / "processed" / "posalpha_estimator_audit.csv",
    ROOT / "results" / "scheduler_overhead_micro" / "processed" / "scheduler_all_summary.csv",
    ROOT / "results" / "scheduler_overhead_micro" / "processed" / "quality_checks.csv",
    ROOT / "results" / "processed" / "route_kv_stress_summary.csv",
    ROOT / "results" / "processed" / "bestfixed_stress_summary.csv",
    ROOT / "results" / "smoke_poisson_openloop" / "processed" / "scheduler_all_summary.csv",
    ROOT / "results" / "smoke_poisson_openloop" / "processed" / "quality_checks.csv",
    ROOT / "results" / "smoke_transfer_gpu" / "processed" / "scheduler_transfer_gpu_summary.csv",
    ROOT / "results" / "smoke_transfer_gpu" / "processed" / "transfer_gpu_bestfixed_summary.csv",
]

REQUIRED_RAW = [
    ROOT / "results" / "local_v3_kcap3" / "raw" / "run_manifest.json",
    ROOT / "results" / "local_v3_kcap3" / "raw" / "effective_config.json",
    ROOT / "results" / "local_v3_kcap3" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_main" / "seed_42" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_main" / "seed_42" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_main" / "seed_42" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_main" / "seed_123" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_main" / "seed_123" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_main" / "seed_123" / "raw" / "run_status.csv",
    ROOT / "results" / "eagle2_compare" / "seed_7" / "raw" / "run_manifest.json",
    ROOT / "results" / "eagle2_compare" / "seed_7" / "raw" / "effective_config.json",
    ROOT / "results" / "eagle2_compare" / "seed_7" / "raw" / "run_status.csv",
    ROOT / "results" / "eagle2_compare" / "seed_42" / "raw" / "run_manifest.json",
    ROOT / "results" / "eagle2_compare" / "seed_42" / "raw" / "effective_config.json",
    ROOT / "results" / "eagle2_compare" / "seed_42" / "raw" / "run_status.csv",
    ROOT / "results" / "eagle2_compare" / "seed_123" / "raw" / "run_manifest.json",
    ROOT / "results" / "eagle2_compare" / "seed_123" / "raw" / "effective_config.json",
    ROOT / "results" / "eagle2_compare" / "seed_123" / "raw" / "run_status.csv",
    ROOT / "results" / "specserve_targeted" / "raw" / "run_manifest.json",
    ROOT / "results" / "specserve_targeted" / "raw" / "effective_config.json",
    ROOT / "results" / "specserve_targeted" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_7" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_7" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_7" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_42" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_42" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_42" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_123" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_123" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_heterog_poisson" / "seed_123" / "raw" / "run_status.csv",
    ROOT / "results" / "evictv2_targeted" / "raw" / "run_manifest.json",
    ROOT / "results" / "evictv2_targeted" / "raw" / "effective_config.json",
    ROOT / "results" / "evictv2_targeted" / "raw" / "run_status.csv",
    ROOT / "results" / "evict_guarded_targeted" / "raw" / "run_manifest.json",
    ROOT / "results" / "evict_guarded_targeted" / "raw" / "effective_config.json",
    ROOT / "results" / "evict_guarded_targeted" / "raw" / "run_status.csv",
    ROOT / "results" / "second_model_targeted" / "raw" / "run_manifest.json",
    ROOT / "results" / "second_model_targeted" / "raw" / "effective_config.json",
    ROOT / "results" / "second_model_targeted" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_7" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_7" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_7" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_42" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_42" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_42" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_123" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_123" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_second_model_n32" / "seed_123" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_7" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_7" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_7" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_42" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_42" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_42" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_123" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_123" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_pythia_model_n32" / "seed_123" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_7" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_7" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_7" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_42" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_42" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_42" / "raw" / "run_status.csv",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_123" / "raw" / "run_manifest.json",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_123" / "raw" / "effective_config.json",
    ROOT / "results" / "tpds_multiseed_edge_practical" / "seed_123" / "raw" / "run_status.csv",
    ROOT / "results" / "verify_latency_fit" / "raw" / "verify_latency_measurements.csv",
    ROOT / "results" / "scheduler_overhead_micro" / "raw" / "run_manifest.json",
    ROOT / "results" / "scheduler_overhead_micro" / "raw" / "effective_config.json",
    ROOT / "results" / "scheduler_overhead_micro" / "raw" / "run_status.csv",
    ROOT / "results" / "raw" / "route_kv_stress_sim" / "route_kv_stress_raw.csv",
    ROOT / "results" / "raw" / "route_kv_stress_sim" / "route_kv_stress_manifest.json",
    ROOT / "results" / "raw" / "bestfixed_stress_sim" / "bestfixed_stress_raw.csv",
    ROOT / "results" / "raw" / "bestfixed_stress_sim" / "bestfixed_stress_manifest.json",
    ROOT / "results" / "smoke_poisson_openloop" / "raw" / "run_manifest.json",
    ROOT / "results" / "smoke_poisson_openloop" / "raw" / "effective_config.json",
    ROOT / "results" / "smoke_poisson_openloop" / "raw" / "run_status.csv",
    ROOT / "results" / "smoke_transfer_gpu" / "raw" / "run_manifest.json",
    ROOT / "results" / "smoke_transfer_gpu" / "raw" / "effective_config.json",
    ROOT / "results" / "smoke_transfer_gpu" / "raw" / "run_status.csv",
]

SUMMARY_REQUIRED_COLUMNS = {
    "tag",
    "kind",
    "policy",
    "N",
    "x_seed",
    "target_model",
    "draft_model",
    "goodput_tok_s",
    "throughput_tok_s",
    "slo_attainment",
    "p99_tpot_ms",
    "kv_evictions_per_1k_tokens",
    "reprefills_per_1k_tokens",
    "configured_beta",
    "configured_kv_blocks",
    "x_max_batch",
    "x_peak_gpu_mb",
}

MULTISEED_REQUIRED_COLUMNS = {
    "N",
    "baseline",
    "goodput_lift_pct_mean",
    "goodput_lift_pct_std",
    "seeds",
}

EXPECTED_MAIN_FIGURES = [
    "figure1_motivation_singlecol.pdf",
    "main_scheduler_overview.pdf",
    "slo_sensitivity_overview.pdf",
    "kv_sweep_overview.pdf",
    "ablation_throughput.pdf",
    "ablation_p99_tpot.pdf",
    "edge_main_bottleneck.pdf",
    "edge_service_breakdown_n8.pdf",
]

EXPECTED_SCIENTIFIC_ROUTE_KV_FIGURES = [
    "route_kv_ablation_scientific.pdf",
    "route_kv_ablation_scientific.png",
    "route_kv_ablation_scientific.svg",
    "route_kv_joint_stress_scientific.pdf",
    "route_kv_joint_stress_scientific.png",
    "route_kv_joint_stress_scientific.svg",
]

EXPECTED_SCIENTIFIC_BESTFIXED_FIGURES = [
    "bestfixed_stress_scientific.pdf",
    "bestfixed_stress_scientific.png",
    "bestfixed_stress_scientific.svg",
]

EXPECTED_SCIENTIFIC_MANUSCRIPT_FIGURES = [
    "edge_main_bottleneck_scientific.pdf",
    "edge_service_breakdown_n8_scientific.pdf",
    "main_scheduler_overview_scientific.pdf",
    "bestfixed_stress_scientific.pdf",
    "slo_sensitivity_overview_scientific.pdf",
    "kv_sweep_overview_scientific.pdf",
    "ablation_throughput_scientific.pdf",
    "ablation_p99_tpot_scientific.pdf",
    "route_kv_ablation_scientific.pdf",
    "route_kv_joint_stress_scientific.pdf",
]

ROUTE_KV_REQUIRED_COLUMNS = {
    "N",
    "kv_blocks",
    "route_delay_ms",
    "routing_mode",
    "goodput_tok_s_mean",
    "p99_tpot_ms_mean",
    "kv_utilization_mean",
    "remote_hit_ratio_mean",
    "migration_rate_mean",
    "routing_oscillation_rate_mean",
    "kv_evictions_per_1k_tokens_mean",
    "reprefills_per_1k_tokens_mean",
    "collapse_rate",
}

BESTFIXED_STRESS_REQUIRED_COLUMNS = {
    "scenario",
    "N",
    "method",
    "selected_policies",
    "goodput_tok_s_mean",
    "p99_tpot_ms_mean",
    "slo_attainment_mean",
    "kv_evictions_per_1k_tokens_mean",
    "recovery_time_s_mean",
}


def record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = "") -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return set(reader.fieldnames or [])


def run_json(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = None
        text = proc.stdout
        decoder = json.JSONDecoder()
        for start in [idx for idx, char in enumerate(text) if char == "{"]:
            try:
                payload, _ = decoder.raw_decode(text[start:])
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            payload = {"status": "parse_error", "output": proc.stdout}
    payload["_returncode"] = proc.returncode
    return payload


def check_paths(checks: list[dict[str, Any]], label: str, paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    record(checks, f"{label} files exist", not missing, missing)


def check_processed_fields(checks: list[dict[str, Any]]) -> None:
    path = ROOT / "results" / "local_v3_kcap3" / "processed" / "scheduler_all_summary.csv"
    rows = read_csv(path) if path.exists() else []
    columns = csv_columns(path) if path.exists() else set()
    missing = sorted(SUMMARY_REQUIRED_COLUMNS - columns)
    record(checks, "scheduler_all_summary has required fields", not missing, missing)
    record(checks, "scheduler_all_summary has rows", len(rows) >= 100, {"rows": len(rows), "expected_min": 100})

    policies = {row.get("policy", "") for row in rows}
    expected_policies = {"NoSpec", "FixedK-1", "FixedK-3", "FixedK-5", "FixedK-7", "IndepOpt", "TETRIS-Proxy", "SpecSched"}
    record(checks, "main policies available for reproduction", expected_policies.issubset(policies), sorted(expected_policies - policies))

    multiseed = ROOT / "results" / "tpds_multiseed_main" / "processed" / "main_multiseed_paired_lifts.csv"
    columns = csv_columns(multiseed) if multiseed.exists() else set()
    missing = sorted(MULTISEED_REQUIRED_COLUMNS - columns)
    record(checks, "multiseed paired lifts have required fields", not missing, missing)

    heterog = ROOT / "results" / "tpds_multiseed_heterog_poisson" / "processed" / "main_multiseed_paired_lifts.csv"
    columns = csv_columns(heterog) if heterog.exists() else set()
    missing = sorted(MULTISEED_REQUIRED_COLUMNS - columns)
    record(checks, "heterogeneous poisson paired lifts have required fields", not missing, missing)

    second_model = ROOT / "results" / "second_model_targeted" / "processed" / "scheduler_all_summary.csv"
    columns = csv_columns(second_model) if second_model.exists() else set()
    missing = sorted(SUMMARY_REQUIRED_COLUMNS - columns)
    record(checks, "second-model summary has required fields", not missing, missing)


def check_focused_audit_artifacts(checks: list[dict[str, Any]]) -> None:
    fit_metrics_path = ROOT / "results" / "verify_latency_fit" / "processed" / "verify_latency_fit_metrics.json"
    perturb_path = ROOT / "results" / "verify_latency_fit" / "processed" / "verify_latency_perturbation.json"
    posalpha_path = ROOT / "results" / "estimator_audit" / "processed" / "posalpha_estimator_audit.json"
    fit_metrics = json.loads(fit_metrics_path.read_text(encoding="utf-8")) if fit_metrics_path.exists() else {}
    fit_values = fit_metrics.get("metrics", fit_metrics)
    perturb = json.loads(perturb_path.read_text(encoding="utf-8")) if perturb_path.exists() else {}
    posalpha = json.loads(posalpha_path.read_text(encoding="utf-8")) if posalpha_path.exists() else {}

    record(
        checks,
        "verify-latency fit artifact is plausible",
        int(fit_values.get("count", 0)) >= 64
        and float(fit_values.get("mae_ms", 99.0)) <= 1.5
        and float(fit_values.get("r2", 0.0)) >= 0.93,
        fit_metrics,
    )
    record(
        checks,
        "verify-latency perturbation artifact is present",
        "max_degradation_pct" in perturb,
        perturb,
    )
    record(
        checks,
        "posalpha estimator audit artifact is present",
        "max_abs_goodput_delta_pct" in posalpha,
        posalpha,
    )


def check_route_kv_stress_artifacts(checks: list[dict[str, Any]]) -> None:
    summary_path = ROOT / "results" / "processed" / "route_kv_stress_summary.csv"
    raw_path = ROOT / "results" / "raw" / "route_kv_stress_sim" / "route_kv_stress_raw.csv"
    manifest_path = ROOT / "results" / "raw" / "route_kv_stress_sim" / "route_kv_stress_manifest.json"
    table_path = PAPER_ASSETS / "tables" / "route_kv_stress_sim.tex"

    summary_rows = read_csv(summary_path) if summary_path.exists() else []
    raw_rows = read_csv(raw_path) if raw_path.exists() else []
    summary_columns = csv_columns(summary_path) if summary_path.exists() else set()
    missing_columns = sorted(ROUTE_KV_REQUIRED_COLUMNS - summary_columns)
    record(checks, "route/KV stress summary has required fields", not missing_columns, missing_columns)
    record(
        checks,
        "route/KV stress simulation row counts are plausible",
        len(raw_rows) >= 480 and len(summary_rows) >= 96,
        {"raw_rows": len(raw_rows), "summary_rows": len(summary_rows)},
    )

    focus = {
        row.get("routing_mode"): row
        for row in summary_rows
        if row.get("N") == "64" and row.get("kv_blocks") == "512" and row.get("route_delay_ms") == "5.0"
    }
    expected_modes = {"local-only", "random", "kv-aware", "oracle"}
    mode_missing = sorted(expected_modes - set(focus))
    kv_aware_gt_local = False
    random_has_oscillation = False
    if not mode_missing:
        kv_aware_gt_local = float(focus["kv-aware"]["goodput_tok_s_mean"]) > float(focus["local-only"]["goodput_tok_s_mean"])
        random_has_oscillation = float(focus["random"]["routing_oscillation_rate_mean"]) > float(
            focus["kv-aware"]["routing_oscillation_rate_mean"]
        )
    record(
        checks,
        "route/KV focus cell contains routing ablation signal",
        not mode_missing and kv_aware_gt_local and random_has_oscillation,
        {
            "missing_modes": mode_missing,
            "kv_aware_gt_local": kv_aware_gt_local,
            "random_has_oscillation": random_has_oscillation,
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    record(
        checks,
        "route/KV manifest scopes data as calibrated simulation",
        "Calibrated simulation" in str(manifest.get("note", "")),
        manifest.get("note", ""),
    )

    table_ok = table_path.exists() and "calibrated simulation" in table_path.read_text(encoding="utf-8").lower()
    record(checks, "route/KV table exists and states simulation scope", table_ok, str(table_path.relative_to(ROOT)))

    missing_figures: list[str] = []
    for root in (PAPER_ASSETS / "figures_scientific", MAIN_DIR / "figures_scientific"):
        for name in EXPECTED_SCIENTIFIC_ROUTE_KV_FIGURES:
            path = root / name
            if not path.exists() or path.stat().st_size == 0:
                missing_figures.append(str(path.relative_to(ROOT)))
    record(checks, "route/KV scientific figures exist in paper assets and manuscript", not missing_figures, missing_figures)


def check_bestfixed_stress_artifacts(checks: list[dict[str, Any]]) -> None:
    summary_path = ROOT / "results" / "processed" / "bestfixed_stress_summary.csv"
    raw_path = ROOT / "results" / "raw" / "bestfixed_stress_sim" / "bestfixed_stress_raw.csv"
    manifest_path = ROOT / "results" / "raw" / "bestfixed_stress_sim" / "bestfixed_stress_manifest.json"
    table_path = PAPER_ASSETS / "tables" / "bestfixed_stress_sim.tex"

    summary_rows = read_csv(summary_path) if summary_path.exists() else []
    raw_rows = read_csv(raw_path) if raw_path.exists() else []
    summary_columns = csv_columns(summary_path) if summary_path.exists() else set()
    missing_columns = sorted(BESTFIXED_STRESS_REQUIRED_COLUMNS - summary_columns)
    record(checks, "BestFixed stress summary has required fields", not missing_columns, missing_columns)
    record(
        checks,
        "BestFixed stress simulation row counts are plausible",
        len(raw_rows) >= 90 and len(summary_rows) >= 18,
        {"raw_rows": len(raw_rows), "summary_rows": len(summary_rows)},
    )

    rows64 = [row for row in summary_rows if row.get("N") == "64"]
    by_cell = {(row.get("scenario"), row.get("method")): row for row in rows64}
    scenarios = {"workload_shift", "bursty_arrival", "tiered_slo"}
    missing_cells = [
        (scenario, method)
        for scenario in scenarios
        for method in ("BestFixed-Global", "SpecSched")
        if (scenario, method) not in by_cell
    ]
    lifts: dict[str, float] = {}
    if not missing_cells:
        for scenario in scenarios:
            spec = float(by_cell[(scenario, "SpecSched")]["goodput_tok_s_mean"])
            fixed = float(by_cell[(scenario, "BestFixed-Global")]["goodput_tok_s_mean"])
            lifts[scenario] = (spec / fixed - 1.0) * 100.0
    record(
        checks,
        "BestFixed stress diagnostic shows transfer gap",
        not missing_cells and all(value >= 10.0 for value in lifts.values()),
        {"missing_cells": missing_cells, "lifts_pct": lifts},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    record(
        checks,
        "BestFixed stress manifest scopes data as calibrated simulation",
        "Calibrated simulation" in str(manifest.get("note", "")),
        manifest.get("note", ""),
    )
    table_ok = table_path.exists() and "calibrated simulation" in table_path.read_text(encoding="utf-8").lower()
    record(checks, "BestFixed stress table exists and states simulation scope", table_ok, str(table_path.relative_to(ROOT)))

    missing_figures: list[str] = []
    for root in (PAPER_ASSETS / "figures_scientific", MAIN_DIR / "figures_scientific"):
        for name in EXPECTED_SCIENTIFIC_BESTFIXED_FIGURES:
            path = root / name
            if not path.exists() or path.stat().st_size == 0:
                missing_figures.append(str(path.relative_to(ROOT)))
    record(checks, "BestFixed stress scientific figures exist in paper assets and manuscript", not missing_figures, missing_figures)


def check_baseline_map_artifact_scope(checks: list[dict[str, Any]]) -> None:
    readme_candidates = [
        ROOT / "main" / "tpds_artifact_README.md",
        ROOT / "README.md",
    ]
    text = ""
    for readme in readme_candidates:
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            break
    required_snippets = [
        "configs/tpds_multiseed_eagle2_compare.json",
        "results/eagle2_compare/",
        "TETRIS-Proxy, EAGLE-2-Proxy, and SpecServe-Proxy",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    record(checks, "artifact README documents baseline-map evidence sources", not missing, missing)

    paired_path = ROOT / "results" / "eagle2_compare" / "processed" / "main_multiseed_paired_lifts.csv"
    rows = read_csv(paired_path) if paired_path.exists() else []
    baselines = {(row.get("N"), row.get("baseline")) for row in rows}
    required_baselines = {
        (n_value, baseline)
        for n_value in ("16", "32", "64")
        for baseline in ("TETRIS-Proxy", "EAGLE-2-Proxy", "SpecServe-Proxy")
    }
    missing_pairs = sorted(required_baselines - baselines)
    record(checks, "artifact contains baseline-map paired lifts", not missing_pairs, missing_pairs)


def check_trace_replay_scaffold(checks: list[dict[str, Any]]) -> None:
    cfg_path = ROOT / "configs" / "smoke_trace_replay_paper.json"
    trace_path = ROOT / "data" / "synthetic_production_trace_smoke.jsonl"
    if not cfg_path.exists() or not trace_path.exists():
        missing = [
            str(path.relative_to(ROOT))
            for path in (cfg_path, trace_path)
            if not path.exists()
        ]
        record(checks, "trace replay scaffold files exist", False, missing)
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    common = cfg.get("common", {})
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    has_arrivals = all(
        any(field in row for field in ("arrival_s", "arrival_time", "timestamp", "time", "start_time"))
        for row in trace_rows
    )
    has_outputs = all(
        any(field in row for field in ("max_new_tokens", "output_length", "completion_tokens", "response_tokens", "generated_tokens"))
        for row in trace_rows
    )
    record(
        checks,
        "trace replay scaffold config is result-neutral",
        common.get("arrival_model") == "trace"
        and str(common.get("arrival_trace", "")).endswith("synthetic_production_trace_smoke.jsonl")
        and has_arrivals
        and has_outputs,
        {
            "arrival_model": common.get("arrival_model"),
            "arrival_trace": common.get("arrival_trace"),
            "trace_rows": len(trace_rows),
            "has_arrivals": has_arrivals,
            "has_outputs": has_outputs,
        },
    )

    doc_groups = [
        [
            ROOT / "main" / "tpds_artifact_README.md",
            ROOT / "README.md",
        ],
        [
            ROOT / "main" / "artifact_reproducibility_tpds.md",
        ],
    ]
    required_doc_snippets = [
        "configs/smoke_trace_replay_paper.json",
        "data/synthetic_production_trace_smoke.jsonl",
        "plumbing",
        "Azure",
        "BurstGPT",
        "WildChat",
        "not",
        "production",
        "performance result",
    ]
    doc_missing: dict[str, list[str]] = {}
    for candidates in doc_groups:
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            doc_missing[str(path.relative_to(ROOT))] = ["missing file"]
            continue
        text = path.read_text(encoding="utf-8")
        missing = [snippet for snippet in required_doc_snippets if snippet not in text]
        if missing:
            doc_missing[str(path.relative_to(ROOT))] = missing
    record(
        checks,
        "artifact docs scope trace replay as scaffold only",
        not doc_missing,
        doc_missing,
    )


def check_transfer_gpu_scaffold(checks: list[dict[str, Any]]) -> None:
    cfg_path = ROOT / "configs" / "paper3090_transfer_gpu.json"
    smoke_cfg_path = ROOT / "configs" / "smoke_transfer_gpu.json"
    generator = ROOT / "experiments" / "real_matrix" / "generate_transfer_traces.py"
    processor = ROOT / "experiments" / "real_matrix" / "process_transfer_gpu_results.py"
    smoke_summary = ROOT / "results" / "smoke_transfer_gpu" / "processed" / "scheduler_transfer_gpu_summary.csv"
    smoke_bestfixed = ROOT / "results" / "smoke_transfer_gpu" / "processed" / "transfer_gpu_bestfixed_summary.csv"
    trace_manifest = ROOT / "data" / "transfer_traces_smoke" / "manifest.json"
    required = [cfg_path, smoke_cfg_path, generator, processor, smoke_summary, smoke_bestfixed, trace_manifest]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        record(checks, "transfer GPU scaffold files exist", False, missing)
        return

    rows = read_csv(smoke_summary)
    bestfixed_rows = read_csv(smoke_bestfixed)
    required_cols = {
        "policy",
        "N",
        "tag",
        "kind",
        "arrival_model",
        "x_arrival_trace",
        "x_request_count",
        "x_seed",
        "goodput_tok_s",
        "p99_tpot_ms",
    }
    row_cols = set(rows[0]) if rows else set()
    manifest = json.loads(trace_manifest.read_text(encoding="utf-8"))
    smoke_tags = {str(row.get("tag", "")) for row in rows}
    record(
        checks,
        "transfer GPU smoke emits expected CSV fields and scenarios",
        bool(rows)
        and required_cols.issubset(row_cols)
        and {"smoke_transfer_shift", "smoke_transfer_bursty", "smoke_transfer_tiered"}.issubset(smoke_tags)
        and bool(bestfixed_rows)
        and "Trace inputs only" in str(manifest.get("scope", "")),
        {
            "rows": len(rows),
            "bestfixed_rows": len(bestfixed_rows),
            "missing_columns": sorted(required_cols - row_cols),
            "tags": sorted(smoke_tags),
            "manifest_scope": manifest.get("scope", ""),
        },
    )

    paper_paths = [ROOT / "main" / "main.tex", ROOT / "main" / "appendix.tex"]
    paper_text = "\n".join(
        path.read_text(encoding="utf-8") for path in paper_paths if path.exists()
    )
    normalized_paper = " ".join(paper_text.split())
    normalized_lower = normalized_paper.lower()
    multiseed_bestfixed = (
        ROOT
        / "results"
        / "tpds_multiseed_transfer_gpu"
        / "processed"
        / "transfer_gpu_multiseed_bestfixed_summary.csv"
    )
    measured_bestfixed = ROOT / "results" / "transfer_gpu" / "processed" / "transfer_gpu_bestfixed_summary.csv"
    if multiseed_bestfixed.exists():
        measured_bestfixed = multiseed_bestfixed
    measured_rows = read_csv(measured_bestfixed) if measured_bestfixed.exists() else []
    measured_scenarios = {str(row.get("scenario", "")) for row in measured_rows}
    measured_caption_ok = (
        "measured seed-7 non-stationary transfer replay at $n=64$" in normalized_lower
        or "measured three-seed non-stationary transfer replay at $n=64$" in normalized_lower
    )
    transfer_scope_terms = (
        measured_caption_ok
        and "diagnostic and are not used as headline gains" in normalized_lower
        and "current measured replay is mixed-to-negative" in normalized_lower
        and "measured diagnostic, not headline evidence" in normalized_lower
        and {"Bursty arrivals", "Workload shift", "Tiered SLOs"}.issubset(measured_scenarios)
    )
    record(
        checks,
        "paper scopes transfer GPU replay as diagnostic rather than headline rows",
        transfer_scope_terms,
        {
            "measured_source": str(measured_bestfixed.relative_to(ROOT)),
            "measured_scenarios": sorted(measured_scenarios),
            "measured_rows": len(measured_rows),
        },
    )


def check_result_level_docker_scope(checks: list[dict[str, Any]]) -> None:
    dockerfile = ROOT / "Dockerfile.artifact"
    if not dockerfile.exists():
        record(checks, "result-level Dockerfile exists", False, str(dockerfile.relative_to(ROOT)))
        return

    text = dockerfile.read_text(encoding="utf-8")
    required_snippets = [
        "FROM python:3.11-slim",
        "matplotlib==3.8.4",
        "pandas==2.2.2",
        'CMD ["make", "tpds-results-reproduce"]',
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    forbidden = [
        "torch",
        "vllm",
        "cuda",
        "transformers",
    ]
    forbidden_hits = [snippet for snippet in forbidden if snippet in text.lower()]
    record(
        checks,
        "result-level Dockerfile is dependency-minimal",
        not missing and not forbidden_hits,
        {"missing": missing, "forbidden_hits": forbidden_hits},
    )

    readme_candidates = [
        ROOT / "main" / "tpds_artifact_README.md",
        ROOT / "README.md",
    ]
    readme_text = ""
    for readme in readme_candidates:
        if readme.exists():
            readme_text = readme.read_text(encoding="utf-8")
            break
    required_readme = [
        "Dockerfile.artifact",
        "does not include CUDA",
        "PyTorch",
        "vLLM",
        "model checkpoints",
        "Jetson access",
        "not a full GPU rerun environment",
    ]
    missing_readme = [snippet for snippet in required_readme if snippet not in readme_text]
    record(checks, "artifact README scopes Docker as result-level only", not missing_readme, missing_readme)


def check_quality_csv(checks: list[dict[str, Any]]) -> None:
    quality_paths = [
        ROOT / "results" / "local_v3_kcap3" / "processed" / "quality_checks.csv",
        ROOT / "results" / "specserve_targeted" / "processed" / "quality_checks.csv",
    ]
    failures: list[dict[str, str]] = []
    for path in quality_paths:
        if not path.exists():
            failures.append({"path": str(path.relative_to(ROOT)), "error": "missing"})
            continue
        for row in read_csv(path):
            if row.get("failed_cells") != "0" or row.get("missing_cells") != "[]" or row.get("goodput_all_zero") != "0":
                failures.append({"path": str(path.relative_to(ROOT)), "tag": row.get("tag", ""), "error": "quality failure"})
    record(checks, "processed quality checks pass", not failures, failures)


def check_plot_manifest(checks: list[dict[str, Any]]) -> None:
    if not MANIFEST.exists():
        record(checks, "plot manifest exists", False, str(MANIFEST.relative_to(ROOT)))
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record(checks, "plot manifest exists", True, str(MANIFEST.relative_to(ROOT)))
    record(checks, "plot manifest row count plausible", int(manifest.get("rows", 0)) >= 100, manifest.get("rows"))

    figures = manifest.get("generated_figures", [])
    tables = manifest.get("generated_tables", [])
    missing_figures = []
    for name in figures:
        path = PAPER_ASSETS / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            missing_figures.append(name)
    missing_tables = []
    for name in tables:
        path = PAPER_ASSETS / "tables" / name
        if not path.exists() or path.stat().st_size == 0:
            missing_tables.append(name)
    record(checks, "manifest figures exist and are non-empty", not missing_figures, missing_figures)
    record(checks, "manifest tables exist and are non-empty", not missing_tables, missing_tables)

    missing_main = []
    for name in EXPECTED_MAIN_FIGURES:
        if (MAIN_DIR / "figures" / name).exists():
            continue
        if (PAPER_ASSETS / "figures" / name).exists():
            continue
        missing_main.append(name)
    record(checks, "paper figure PDFs exist", not missing_main, missing_main)

    missing_scientific = []
    for name in EXPECTED_SCIENTIFIC_MANUSCRIPT_FIGURES:
        paper_path = PAPER_ASSETS / "figures_scientific" / name
        main_path = MAIN_DIR / "figures_scientific" / name
        if not paper_path.exists() or paper_path.stat().st_size == 0:
            missing_scientific.append(str(paper_path.relative_to(ROOT)))
        if not main_path.exists() or main_path.stat().st_size == 0:
            missing_scientific.append(str(main_path.relative_to(ROOT)))
    record(checks, "scientific manuscript figures exist in assets and main", not missing_scientific, missing_scientific)


def check_claim_audit(checks: list[dict[str, Any]]) -> None:
    payload = run_json(["python", "scripts/audit_paper_claims.py"])
    record(
        checks,
        "headline claim audit passes",
        payload.get("_returncode") == 0 and payload.get("status") == "pass",
        {"status": payload.get("status"), "failed": payload.get("failed"), "returncode": payload.get("_returncode")},
    )


def check_upload_exclusion(checks: list[dict[str, Any]]) -> None:
    if not (ROOT / "scripts" / "verify_tpds_package.py").exists():
        record(checks, "formal upload excludes local artifact notes", True, "skipped in artifact-only bundle")
        return
    payload = run_json(["python", "scripts/verify_tpds_package.py"])
    names = {check.get("name"): check for check in payload.get("checks", [])}
    local_only = names.get("local-only helper files excluded", {})
    record(
        checks,
        "formal upload excludes local artifact notes",
        bool(local_only.get("pass")),
        {"package_status": payload.get("status"), "local_only_check": local_only.get("detail")},
    )


def main() -> int:
    checks: list[dict[str, Any]] = []
    check_paths(checks, "config", REQUIRED_CONFIGS)
    check_paths(checks, "processed artifact", REQUIRED_PROCESSED)
    check_paths(checks, "raw artifact", REQUIRED_RAW)
    check_processed_fields(checks)
    check_focused_audit_artifacts(checks)
    check_route_kv_stress_artifacts(checks)
    check_bestfixed_stress_artifacts(checks)
    check_baseline_map_artifact_scope(checks)
    check_trace_replay_scaffold(checks)
    check_transfer_gpu_scaffold(checks)
    check_result_level_docker_scope(checks)
    check_quality_csv(checks)
    check_plot_manifest(checks)
    check_claim_audit(checks)
    check_upload_exclusion(checks)

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

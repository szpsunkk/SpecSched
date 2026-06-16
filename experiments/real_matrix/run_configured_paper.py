"""Run the ex.md paper experiment matrix from a JSON config.

The runner deliberately starts one scheduler subprocess per (experiment, N)
cell. Policies at the same N share one NoSpec SLO calibration, while each N is
isolated from long-lived CUDA allocator state.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from .config_utils import (
        config_file_hash,
        hyperparam_hash,
        load_config as load_config_file,
        runtime_config_hash,
    )
except ImportError:
    from config_utils import (
        config_file_hash,
        hyperparam_hash,
        load_config as load_config_file,
        runtime_config_hash,
    )


ROOT = Path(__file__).resolve().parents[2]


def ensure_dirs(results_root: Path) -> Dict[str, Path]:
    paths = {
        "raw": results_root / "raw",
        "processed": results_root / "processed",
        "figures": results_root / "figures",
        "logs": results_root / "raw" / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def csv_list(values: Iterable) -> str:
    return ",".join(str(v) for v in values)


def value_for_n(value, n_value: int, default=None):
    """Return a scalar config value, allowing optional per-N mappings."""
    if value is None:
        return default
    if isinstance(value, dict):
        for key in (str(n_value), f"N{n_value}", n_value):
            if key in value:
                return value[key]
        if "default" in value:
            return value["default"]
        return default
    return value


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def build_command(cfg: dict, exp: dict, n_value: int, out_dir: Path) -> List[str]:
    common = {**cfg["common"], **exp.get("common_overrides", {})}
    models = cfg["models"]
    paths = cfg["paths"]
    labels = f"{models['draft_label']}_to_{models['target_label']}"
    tag = exp["tag"]
    cell = f"N{n_value}"
    py = cfg["execution"].get("python", "python")
    requests_per_n = value_for_n(
        exp.get("requests_per_N", common.get("requests_per_N")),
        n_value,
        common.get("requests_per_N"),
    )
    arrival_rate_rps = value_for_n(
        exp.get("arrival_rate_rps", common.get("arrival_rate_rps", 0.0)),
        n_value,
        common.get("arrival_rate_rps", 0.0),
    )
    arrival_trace = value_for_n(
        exp.get("arrival_trace", common.get("arrival_trace", "")),
        n_value,
        common.get("arrival_trace", ""),
    )
    burst_size = value_for_n(
        exp.get("burst_size", common.get("burst_size", 4)),
        n_value,
        common.get("burst_size", 4),
    )
    burst_period_s = value_for_n(
        exp.get("burst_period_s", common.get("burst_period_s", 1.0)),
        n_value,
        common.get("burst_period_s", 1.0),
    )
    tier_names = common.get("tier_names", exp.get("tier_names", []))
    tier_weights = common.get("tier_weights", exp.get("tier_weights", []))
    tier_tpot_mults = common.get("tier_tpot_mults", exp.get("tier_tpot_mults", []))
    tier_ttft_mults = common.get("tier_ttft_mults", exp.get("tier_ttft_mults", []))
    max_admit_per_step = value_for_n(
        exp.get("max_admit_per_step", common.get("max_admit_per_step", 0)),
        n_value,
        common.get("max_admit_per_step", 0),
    )

    cmd = [
        py,
        "experiments/scheduler/run_scheduler.py",
        "--target", models["target"],
        "--trace", paths["trace"],
        "--draft-backend", models.get("draft_backend", "local"),
        "--Ns", str(n_value),
        "--requests-per-N", str(requests_per_n),
        "--calib-requests", str(common["calib_requests"]),
        "--calib-max-new-tokens", str(common.get("calib_max_new_tokens", 0)),
        "--prompt-hard-max", str(exp["prompt_hard_max"]),
        "--max-new-tokens", str(exp["max_new_tokens"]),
        "--output-lengths", csv_list(exp["output_lengths"]),
        "--fixed-ks", csv_list(common["fixed_ks"]),
        "--kmax", str(common["kmax"]),
        "--max-batch", str(common["max_batch"]),
        "--max-admit-per-step", str(max_admit_per_step),
        "--backpressure-pressure-threshold", str(common.get("backpressure_pressure_threshold", 2.0)),
        "--backpressure-delay-threshold-s", str(common.get("backpressure_delay_threshold_s", 0.25)),
        "--backpressure-cost", str(common.get("backpressure_cost", 0.45)),
        "--backpressure-alpha-floor", str(common.get("backpressure_alpha_floor", 0.78)),
        "--backpressure-urgency-cap", str(common.get("backpressure_urgency_cap", 1.15)),
        "--backpressure-opportunistic-k", str(common.get("backpressure_opportunistic_k", 2)),
        "--backpressure-opportunistic-alpha", str(common.get("backpressure_opportunistic_alpha", 0.78)),
        "--backpressure-opportunistic-pressure", str(common.get("backpressure_opportunistic_pressure", 0.25)),
        "--backpressure-opportunistic-delay-ratio", str(common.get("backpressure_opportunistic_delay_ratio", 0.25)),
        "--backpressure-opportunistic-kv-pressure", str(common.get("backpressure_opportunistic_kv_pressure", 0.70)),
        "--backpressure-fastpath-kv-pressure", str(common.get("backpressure_fastpath_kv_pressure", 0.88)),
        "--pressure-aimd-pressure-threshold", str(common.get("pressure_aimd_pressure_threshold", 1.0)),
        "--pressure-aimd-delay-threshold-s", str(common.get("pressure_aimd_delay_threshold_s", 0.25)),
        "--pressure-aimd-add", str(common.get("pressure_aimd_add", 0.20)),
        "--pressure-aimd-mult", str(common.get("pressure_aimd_mult", 0.70)),
        "--pressure-aimd-min", str(common.get("pressure_aimd_min", 1.0)),
        "--pressure-aimd-max", str(common.get("pressure_aimd_max", 3.0)),
        "--pressure-aimd-recovery-pressure", str(common.get("pressure_aimd_recovery_pressure", 0.25)),
        "--pressure-aimd-recovery-delay-ratio", str(common.get("pressure_aimd_recovery_delay_ratio", 0.25)),
        "--pressure-aimd-delay-gain", str(common.get("pressure_aimd_delay_gain", 0.35)),
        "--pressure-aimd-cost", str(common.get("pressure_aimd_cost", 0.30)),
        "--pressure-aimd-alpha-floor", str(common.get("pressure_aimd_alpha_floor", 0.70)),
        "--pressure-aimd-extra-budget-frac", str(common.get("pressure_aimd_extra_budget_frac", 0.50)),
        "--pressure-aimd-stable-pressure", str(common.get("pressure_aimd_stable_pressure", 0.25)),
        "--pressure-aimd-stable-delay-ratio", str(common.get("pressure_aimd_stable_delay_ratio", 0.25)),
        "--pressure-aimd-active-pressure", str(common.get("pressure_aimd_active_pressure", 2.0)),
        "--pressure-aimd-min-extra-gain", str(common.get("pressure_aimd_min_extra_gain", 0.85)),
        "--speculation-ok-margin", str(common.get("speculation_ok_margin", 0.0)),
        "--cache-replay-chunk-tokens", str(common.get("cache_replay_chunk_tokens", 32)),
        "--log-every-steps", str(common.get("log_every_steps", 0)),
        "--kv-blocks", str(exp["kv_blocks"]),
        "--block-tokens", str(common["block_tokens"]),
        "--arrival-model", str(exp.get("arrival_model", common.get("arrival_model", "closed_loop"))),
        "--arrival-rate-rps", str(arrival_rate_rps),
        "--arrival-trace", str(arrival_trace),
        "--burst-size", str(burst_size),
        "--burst-period-s", str(burst_period_s),
        "--tier-names", csv_list(tier_names),
        "--tier-weights", csv_list(tier_weights),
        "--tier-tpot-mults", csv_list(tier_tpot_mults),
        "--tier-ttft-mults", csv_list(tier_ttft_mults),
        "--slo", "adaptive",
        "--adaptive-slo-tpot-mult", str(exp.get("beta", common["adaptive_slo_tpot_mult"])),
        "--adaptive-slo-ttft-mult", str(common["adaptive_slo_ttft_mult"]),
        "--adaptive-slo-scope", str(common["adaptive_slo_scope"]),
        "--policies", csv_list(exp["policies"]),
        "--seed", str(cfg["experiment"]["seed"]),
        "--out", str(out_dir / f"scheduler_{tag}_{cell}_{labels}.csv"),
        "--request-out", str(out_dir / f"scheduler_requests_{tag}_{cell}_{labels}.csv"),
        "--calib-out", str(out_dir / f"scheduler_calib_{tag}_{cell}_{labels}.csv"),
        "--slo-thresholds-out", str(out_dir / f"slo_thresholds_{tag}_{cell}_{labels}.csv"),
        "--nospec-request-out", str(out_dir / f"nospec_tpot_by_request_{tag}_{cell}_{labels}.csv"),
    ]
    if models.get("draft_backend", "local") == "edge":
        cmd.extend([
            "--draft-remote", models["draft_remote"],
            "--edge-hosts", models["edge_hosts"],
            "--edge-user", models["edge_user"],
            "--edge-password", models["edge_password"],
        ])
        if "edge_request_timeout_s" in models:
            cmd.extend(["--edge-request-timeout-s", str(models["edge_request_timeout_s"])])
    else:
        cmd.extend(["--draft", models["draft"]])
    if common.get("deterministic", False):
        cmd.append("--deterministic")
    return cmd


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def legacy_roots(cfg: dict) -> List[Path]:
    return [Path(p).resolve() for p in cfg.get("paths", {}).get("legacy_results_roots", [])]


def scheduler_candidates(cfg: dict, tag_dir: Path, tag: str) -> List[Path]:
    paths = list(tag_dir.glob(f"scheduler_{tag}_*.csv"))
    for root in legacy_roots(cfg):
        paths.extend(root.glob(f"scheduler_{tag}_*.csv"))
    return [
        p for p in paths
        if not p.name.startswith("scheduler_requests_")
        and not p.name.startswith("scheduler_calib_")
    ]


def _config_mismatches(row: dict, cfg: dict, exp: dict, n_value: int) -> List[Tuple[str, str, str]]:
    common = {**cfg["common"], **exp.get("common_overrides", {})}
    expected_max_batch = 1 if row.get("policy") == "SpecSched-NoBatch" else common["max_batch"]
    expected_arrival_model = str(exp.get("arrival_model", common.get("arrival_model", "closed_loop")))
    expected_arrival_rate = value_for_n(
        exp.get("arrival_rate_rps", common.get("arrival_rate_rps", 0.0)),
        n_value,
        common.get("arrival_rate_rps", 0.0),
    )
    expected_arrival_trace = str(value_for_n(
        exp.get("arrival_trace", common.get("arrival_trace", "")),
        n_value,
        common.get("arrival_trace", ""),
    ))
    expected_burst_size = value_for_n(
        exp.get("burst_size", common.get("burst_size", 4)),
        n_value,
        common.get("burst_size", 4),
    )
    expected_burst_period_s = value_for_n(
        exp.get("burst_period_s", common.get("burst_period_s", 1.0)),
        n_value,
        common.get("burst_period_s", 1.0),
    )
    expected_tier_names = csv_list(common.get("tier_names", exp.get("tier_names", [])))
    expected_request_count = (
        n_value
        if expected_arrival_model == "closed_loop"
        else max(
            n_value,
            int(float(value_for_n(
                exp.get("requests_per_N", common.get("requests_per_N")),
                n_value,
                common.get("requests_per_N"),
            ))),
        )
    )
    checks = {
        "x_max_batch": expected_max_batch,
        "x_kmax": common["kmax"],
        "x_kv_blocks": exp["kv_blocks"],
        "x_block_tokens": common["block_tokens"],
        "x_deterministic": int(bool(common.get("deterministic", False))),
        "x_request_count": expected_request_count,
    }
    mismatches: List[Tuple[str, str, str]] = []
    for key, expected in checks.items():
        if key not in row or row[key] in ("", None):
            mismatches.append((key, str(row.get(key, "")), str(expected)))
            continue
        try:
            if int(float(row[key])) != int(expected):
                mismatches.append((key, str(row[key]), str(expected)))
        except ValueError:
            mismatches.append((key, str(row[key]), str(expected)))
    if str(row.get("arrival_model", "")) != expected_arrival_model:
        mismatches.append(("arrival_model", str(row.get("arrival_model", "")), expected_arrival_model))
    if str(row.get("x_tier_names", "")) != expected_tier_names:
        mismatches.append(("x_tier_names", str(row.get("x_tier_names", "")), expected_tier_names))
    if expected_arrival_model == "trace":
        if str(row.get("x_arrival_trace", "")) != expected_arrival_trace:
            mismatches.append(("x_arrival_trace", str(row.get("x_arrival_trace", "")), expected_arrival_trace))
    elif expected_arrival_model == "bursty":
        burst_size_value = row.get("x_burst_size", row.get("burst_size", ""))
        burst_period_value = row.get("x_burst_period_s", row.get("burst_period_s", ""))
        try:
            if int(float(burst_size_value)) != int(expected_burst_size):
                mismatches.append(("x_burst_size", str(burst_size_value), str(expected_burst_size)))
        except ValueError:
            mismatches.append(("x_burst_size", str(burst_size_value), str(expected_burst_size)))
        try:
            if abs(float(burst_period_value) - float(expected_burst_period_s)) > 1e-9:
                mismatches.append(("x_burst_period_s", str(burst_period_value), str(expected_burst_period_s)))
        except ValueError:
            mismatches.append(("x_burst_period_s", str(burst_period_value), str(expected_burst_period_s)))
    elif "arrival_rate_rps" not in row or row.get("arrival_rate_rps") in ("", None):
        mismatches.append(("arrival_rate_rps", str(row.get("arrival_rate_rps", "")), str(expected_arrival_rate)))
    else:
        try:
            if abs(float(row["arrival_rate_rps"]) - float(expected_arrival_rate)) > 1e-9:
                mismatches.append(("arrival_rate_rps", str(row["arrival_rate_rps"]), str(expected_arrival_rate)))
        except ValueError:
            mismatches.append(("arrival_rate_rps", str(row["arrival_rate_rps"]), str(expected_arrival_rate)))
    if row.get("policy") == "SpecSched-PressureAIMD":
        expected_gain = float(common.get("pressure_aimd_min_extra_gain", 0.85))
        key = "x_pressure_aimd_min_extra_gain"
        if key not in row or row.get(key) in ("", None):
            mismatches.append((key, str(row.get(key, "")), str(expected_gain)))
        else:
            try:
                if abs(float(row[key]) - expected_gain) > 1e-9:
                    mismatches.append((key, str(row[key]), str(expected_gain)))
            except ValueError:
                mismatches.append((key, str(row[key]), str(expected_gain)))
    if row.get("policy") == "SpecSched-Safe":
        expected_margin = float(common.get("speculation_ok_margin", 0.0))
        key = "x_speculation_ok_margin"
        if key not in row or row.get(key) in ("", None):
            mismatches.append((key, str(row.get(key, "")), str(expected_margin)))
        else:
            try:
                if abs(float(row[key]) - expected_margin) > 1e-9:
                    mismatches.append((key, str(row[key]), str(expected_margin)))
            except ValueError:
                mismatches.append((key, str(row[key]), str(expected_margin)))
    return mismatches


def _same_config(row: dict, cfg: dict, exp: dict, n_value: int) -> bool:
    return not _config_mismatches(row, cfg, exp, n_value)


def find_existing_policies(cfg: dict, exp: dict, n_value: int, tag_dir: Path) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for path in scheduler_candidates(cfg, tag_dir, exp["tag"]):
        for row in read_csv_rows(path):
            try:
                row_n = int(float(row.get("N", "")))
            except ValueError:
                continue
            policy = str(row.get("policy", ""))
            if row_n == n_value and policy in exp["policies"] and _same_config(row, cfg, exp, n_value):
                found.setdefault(policy, {"source_path": str(path), "source_priority": 0 if tag_dir in path.parents else 1})
    return found


def find_incompatible_existing(cfg: dict, exp: dict, n_value: int, tag_dir: Path) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for path in scheduler_candidates(cfg, tag_dir, exp["tag"]):
        for row in read_csv_rows(path):
            try:
                row_n = int(float(row.get("N", "")))
            except ValueError:
                continue
            policy = str(row.get("policy", ""))
            if row_n != n_value or policy not in exp["policies"]:
                continue
            mismatches = _config_mismatches(row, cfg, exp, n_value)
            if mismatches:
                found.setdefault(policy, {
                    "source_path": str(path),
                    "source_priority": 0 if tag_dir in path.parents else 1,
                    "mismatch": ";".join(f"{key}:{old}->{new}" for key, old, new in mismatches),
                })
    return found


def existing_cell_complete(cfg: dict, exp: dict, n_value: int, tag_dir: Path) -> bool:
    return set(exp["policies"]).issubset(set(find_existing_policies(cfg, exp, n_value, tag_dir)))


def missing_policies(cfg: dict, exp: dict, n_value: int, tag_dir: Path) -> List[str]:
    found = set(find_existing_policies(cfg, exp, n_value, tag_dir))
    return [policy for policy in exp["policies"] if policy not in found]


def rotate_status_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    rotated = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    path.rename(rotated)
    return rotated


def iter_cells(cfg: dict):
    for exp in cfg["experiments"]:
        for n_value in exp["Ns"]:
            yield exp, int(n_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper3090.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    loaded = load_config_file(config_path, ROOT)
    cfg = loaded.cfg
    raw_config = loaded.raw_text
    results_root = Path(cfg["paths"]["results_root"]).resolve()
    dirs = ensure_dirs(results_root)
    run_status_jsonl = dirs["raw"] / "run_status.jsonl"
    run_status_csv = dirs["raw"] / "run_status.csv"
    manifest = {
        "config": str(config_path),
        "config_file_hash": config_file_hash(raw_config),
        "runtime_config_hash": runtime_config_hash(cfg),
        "hyperparam_hash": hyperparam_hash(cfg),
        "seed": cfg["experiment"]["seed"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "isolate_by": cfg["execution"].get("isolate_by", "experiment_n"),
        "raw_dir": str(dirs["raw"]),
        "processed_dir": str(dirs["processed"]),
        "figures_dir": str(dirs["figures"]),
    }
    if not args.dry_run:
        (dirs["raw"] / "effective_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (dirs["raw"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rotated = rotate_status_file(run_status_jsonl)
        if rotated:
            print(f"[paper-config] archived previous status {rotated}", flush=True)

    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", str(ROOT))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = cfg.get("cuda", {}).get(
        "pytorch_cuda_alloc_conf",
        env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
    cublas_workspace = cfg.get("cuda", {}).get("cublas_workspace_config")
    if cublas_workspace:
        env["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_workspace)
    elif "cublas_workspace_config" in cfg.get("cuda", {}):
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)

    rows: List[dict] = []
    total = sum(1 for _ in iter_cells(cfg))
    for idx, (exp, n_value) in enumerate(iter_cells(cfg), start=1):
        tag_dir = dirs["raw"] / exp["tag"]
        tag_dir.mkdir(parents=True, exist_ok=True)
        cell = f"N{n_value}"
        log_path = dirs["logs"] / f"{exp['tag']}_{cell}.log"
        cmd = build_command(cfg, exp, n_value, tag_dir)
        found = find_existing_policies(cfg, exp, n_value, tag_dir)
        incompatible = find_incompatible_existing(cfg, exp, n_value, tag_dir)
        policies_to_run = missing_policies(cfg, exp, n_value, tag_dir)
        force_full_cell = bool(policies_to_run and incompatible)
        if force_full_cell:
            policies_to_run = list(exp["policies"])
        row = {
            "index": idx,
            "total": total,
            "experiment": exp["name"],
            "kind": exp["kind"],
            "tag": exp["tag"],
            "N": n_value,
            "policies": csv_list(exp["policies"]),
            "kv_blocks": exp["kv_blocks"],
            "beta": exp.get("beta", ""),
            "seed": cfg["experiment"]["seed"],
            "runtime_config_hash": runtime_config_hash(cfg),
            "hyperparam_hash": hyperparam_hash(cfg),
            "log": str(log_path),
            "command": " ".join(cmd),
            "resume_reason": "config_mismatch_full_rerun" if force_full_cell else "",
        }
        for policy in exp["policies"]:
            row[f"source_path_{policy}"] = found.get(policy, {}).get("source_path", "")
            row[f"source_priority_{policy}"] = found.get(policy, {}).get("source_priority", "")
            row[f"config_mismatch_{policy}"] = incompatible.get(policy, {}).get("mismatch", "")
        print(
            f"[paper-config] {idx}/{total} tag={exp['tag']} N={n_value} "
            f"policies={csv_list(exp['policies'])}",
            flush=True,
        )
        if force_full_cell:
            details = ", ".join(f"{policy}({meta['mismatch']})" for policy, meta in sorted(incompatible.items()))
            print(
                f"[skip] tag={exp['tag']} N={n_value} config mismatch in existing rows: {details}; "
                "will re-run all policies for this N",
                flush=True,
            )
        if args.dry_run:
            row.update({"status": "dry_run", "exit_code": ""})
            print("  " + row["command"], flush=True)
        elif cfg["execution"].get("skip_existing", False) and not policies_to_run:
            row.update({"status": "skipped_existing", "exit_code": 0})
        else:
            if cfg["execution"].get("skip_existing", False) and policies_to_run != exp["policies"]:
                cmd = build_command({**cfg, "experiments": cfg["experiments"]}, {**exp, "policies": policies_to_run}, n_value, tag_dir)
                row["policies"] = csv_list(policies_to_run)
                row["command"] = " ".join(cmd)
            t0 = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"[command] {row['command']}\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            row.update({
                "status": "ok" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            })
        rows.append(row)
        if not args.dry_run:
            append_jsonl(run_status_jsonl, row)
            write_csv(run_status_csv, rows)
        if row["status"] == "failed" and not cfg["execution"].get("continue_on_error", True):
            sys.exit(1)
    if args.dry_run:
        print(f"[paper-config] dry-run complete; no files written", flush=True)
    else:
        print(f"[paper-config] wrote {run_status_csv}", flush=True)


if __name__ == "__main__":
    main()

"""Aggregate configured paper raw CSVs into processed paper outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from .config_utils import load_config as load_config_file
except ImportError:
    from config_utils import load_config as load_config_file


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    return load_config_file(path, ROOT).cfg


def numeric(value):
    if value is None or value == "":
        return value
    try:
        text = str(value)
        if "." in text or "e" in text.lower() or text.lower() in ("nan", "inf"):
            return float(text)
        return int(text)
    except ValueError:
        return value


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: numeric(v) for k, v in row.items()} for row in csv.DictReader(f)]


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
            writer.writerow({key: row.get(key, "") for key in fields})


def experiments_by_tag(cfg: dict) -> Dict[str, dict]:
    return {exp["tag"]: exp for exp in cfg["experiments"]}


def value_for_n(value, n_value: int, default=None):
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


def csv_list(values: Iterable) -> str:
    return ",".join(str(v) for v in values)


def collect_scheduler(raw: Path, cfg: dict) -> List[dict]:
    by_tag = experiments_by_tag(cfg)
    by_cell: Dict[tuple, dict] = {}
    sources = [(0, path) for path in sorted(raw.glob("*/scheduler_*.csv"))]
    for root_text in cfg.get("paths", {}).get("legacy_results_roots", []):
        root = Path(root_text).resolve()
        sources.extend((1, path) for path in sorted(root.glob("scheduler_*.csv")))
    for priority, path in sources:
        name = path.name
        if name.startswith("scheduler_requests_") or name.startswith("scheduler_calib_"):
            continue
        for tag, exp in by_tag.items():
            if f"scheduler_{tag}_" not in name:
                continue
            for row in read_csv(path):
                if not _same_config(row, cfg, exp):
                    continue
                cell = (tag, int(float(row.get("N", -1))), str(row.get("policy", "")))
                if cell in by_cell and by_cell[cell].get("source_priority", 9) <= priority:
                    continue
                row["tag"] = tag
                row["experiment"] = exp["name"]
                row["kind"] = exp["kind"]
                row["configured_beta"] = exp.get("beta", "")
                row["configured_kv_blocks"] = exp.get("kv_blocks", "")
                row["source_file"] = str(path)
                row["source_priority"] = priority
                by_cell[cell] = row
            break
    return list(by_cell.values())


def _same_config(row: dict, cfg: dict, exp: dict) -> bool:
    common = {**cfg["common"], **exp.get("common_overrides", {})}
    try:
        n_value = int(float(row.get("N", "")))
    except ValueError:
        return False
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
    for key, expected in checks.items():
        if key not in row or row[key] in ("", None):
            return False
        try:
            if int(float(row[key])) != int(expected):
                return False
        except ValueError:
            return False
    if str(row.get("arrival_model", "")) != expected_arrival_model:
        return False
    if str(row.get("x_tier_names", "")) != expected_tier_names:
        return False
    if expected_arrival_model == "trace":
        if str(row.get("x_arrival_trace", "")) != expected_arrival_trace:
            return False
    elif expected_arrival_model == "bursty":
        try:
            if int(float(row.get("x_burst_size", row.get("burst_size", "")))) != int(expected_burst_size):
                return False
            if abs(float(row.get("x_burst_period_s", row.get("burst_period_s", ""))) - float(expected_burst_period_s)) > 1e-9:
                return False
        except ValueError:
            return False
    else:
        if row.get("arrival_rate_rps") in ("", None):
            return False
        try:
            if abs(float(row["arrival_rate_rps"]) - float(expected_arrival_rate)) > 1e-9:
                return False
        except ValueError:
            return False
    if row.get("policy") == "SpecSched-PressureAIMD":
        key = "x_pressure_aimd_min_extra_gain"
        if key not in row or row.get(key) in ("", None):
            return False
        try:
            if abs(float(row[key]) - float(common.get("pressure_aimd_min_extra_gain", 0.85))) > 1e-9:
                return False
        except ValueError:
            return False
    return True


def named_key(prefix: str, tag: str, row: dict) -> tuple:
    n_value = str(row.get("N", ""))
    if prefix == "slo_thresholds":
        return (tag, n_value, str(row.get("request_id", "")), str(row.get("class", "")), str(row.get("beta", "")))
    if prefix == "scheduler_requests":
        return (tag, n_value, str(row.get("policy", "")), str(row.get("req_id", "")), str(row.get("bucket", "")))
    return (tag, n_value, str(row.get("req_id", "")), str(row.get("bucket", "")))


def configured_tag_n(cfg: dict) -> set:
    pairs = set()
    for exp in cfg["experiments"]:
        for n_value in exp["Ns"]:
            pairs.add((str(exp["tag"]), str(int(n_value))))
    return pairs


def collect_named(raw: Path, prefix: str, cfg: dict, allowed_tag_n: set) -> List[dict]:
    by_tag = experiments_by_tag(cfg)
    rows_by_key: Dict[tuple, dict] = {}
    paths = [(0, path) for path in sorted(raw.glob(f"*/{prefix}_*.csv"))]
    for root_text in cfg.get("paths", {}).get("legacy_results_roots", []):
        paths.extend((1, path) for path in sorted(Path(root_text).resolve().glob(f"{prefix}_*.csv")))
    for priority, path in paths:
        for tag, exp in by_tag.items():
            if f"{prefix}_{tag}_" not in path.name:
                continue
            for row in read_csv(path):
                n_value = str(row.get("N", ""))
                if (tag, n_value) not in allowed_tag_n:
                    continue
                key = named_key(prefix, tag, row)
                if key in rows_by_key and rows_by_key[key].get("source_priority", 9) <= priority:
                    continue
                row["tag"] = tag
                row["experiment"] = exp["name"]
                row["kind"] = exp["kind"]
                row["source_file"] = str(path)
                row["source_priority"] = priority
                rows_by_key[key] = row
            break
    return list(rows_by_key.values())


def is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def tier_fairness_summary(summary_rows: List[dict], request_rows: List[dict]) -> List[dict]:
    wall_by_cell = {
        (str(r.get("tag")), str(r.get("N")), str(r.get("policy"))): as_float(r.get("wall_s"), 0.0)
        for r in summary_rows
    }
    grouped: Dict[tuple, List[dict]] = {}
    for row in request_rows:
        key = (
            str(row.get("tag")),
            str(row.get("N")),
            str(row.get("policy")),
            str(row.get("tier", "default") or "default"),
        )
        grouped.setdefault(key, []).append(row)

    per_tier_rows: List[dict] = []
    goodput_by_cell: Dict[tuple, List[float]] = {}
    for (tag, n_value, policy, tier), rows in sorted(grouped.items()):
        wall_s = wall_by_cell.get((tag, n_value, policy), 0.0)
        generated = sum(as_float(r.get("generated"), 0.0) for r in rows)
        attained_tokens = 0.0
        attained_requests = 0
        for row in rows:
            ttft = as_float(row.get("ttft_ms"), float("inf"))
            tpot = as_float(row.get("tpot_ms"), float("inf"))
            slo_ttft = as_float(row.get("slo_ttft_ms"), float("nan"))
            slo_tpot = as_float(row.get("slo_tpot_ms"), float("nan"))
            ok = True
            if is_number(slo_ttft):
                ok = ok and ttft <= slo_ttft
            if is_number(slo_tpot):
                ok = ok and tpot <= slo_tpot
            if ok:
                attained_requests += 1
                attained_tokens += as_float(row.get("generated"), 0.0)
        goodput = attained_tokens / wall_s if wall_s > 0 else 0.0
        cell_key = (tag, n_value, policy)
        goodput_by_cell.setdefault(cell_key, []).append(goodput)
        per_tier_rows.append({
            "tag": tag,
            "N": n_value,
            "policy": policy,
            "tier": tier,
            "requests": len(rows),
            "tokens": generated,
            "attained_requests": attained_requests,
            "slo_attainment": attained_requests / max(1, len(rows)),
            "goodput_tok_s": goodput,
            "wall_s": wall_s,
        })

    jain_rows: List[dict] = []
    for (tag, n_value, policy), values in sorted(goodput_by_cell.items()):
        denom = len(values) * sum(v * v for v in values)
        jain = (sum(values) ** 2 / denom) if denom > 0 else 0.0
        jain_rows.append({
            "tag": tag,
            "N": n_value,
            "policy": policy,
            "tier": "__jain__",
            "requests": "",
            "tokens": "",
            "attained_requests": "",
            "slo_attainment": "",
            "goodput_tok_s": "",
            "wall_s": wall_by_cell.get((tag, n_value, policy), ""),
            "jain_goodput": jain,
            "tiers": len(values),
        })
    return per_tier_rows + jain_rows


def expected_cells(cfg: dict) -> List[dict]:
    rows = []
    for exp in cfg["experiments"]:
        for n_value in exp["Ns"]:
            for policy in exp["policies"]:
                rows.append({
                    "tag": exp["tag"],
                    "kind": exp["kind"],
                    "N": int(n_value),
                    "policy": policy,
                })
    return rows


def quality_checks(all_rows: List[dict], cfg: dict, run_status: List[dict]) -> List[dict]:
    observed = {(str(r.get("tag")), int(r.get("N")), str(r.get("policy"))) for r in all_rows if r.get("N") != ""}
    failed = set()
    for r in run_status:
        if r.get("status") != "failed":
            continue
        tag = str(r.get("tag"))
        n_value = int(r.get("N"))
        for policy in str(r.get("policies", "")).split(","):
            if policy:
                failed.add((tag, n_value, policy))
    out: List[dict] = []
    for exp in cfg["experiments"]:
        selected = [r for r in all_rows if r.get("tag") == exp["tag"]]
        expected = [(exp["tag"], int(n), pol) for n in exp["Ns"] for pol in exp["policies"]]
        missing = [f"N={n}:{pol}" for _, n, pol in expected if (exp["tag"], n, pol) not in observed]
        goodputs = [as_float(r.get("goodput_tok_s")) for r in selected if is_number(r.get("goodput_tok_s"))]
        attain = [as_float(r.get("slo_attainment")) for r in selected if is_number(r.get("slo_attainment"))]
        evictions = [as_float(r.get("kv_evictions")) for r in selected if is_number(r.get("kv_evictions"))]
        rhos = [as_float(r.get("verifier_load_rho")) for r in selected if is_number(r.get("verifier_load_rho"))]
        indep = [as_float(r.get("mean_selected_k")) for r in selected if r.get("policy") == "IndepOpt" and is_number(r.get("mean_selected_k"))]
        unit_ok = 1
        for row in selected:
            p99 = as_float(row.get("p99_tpot_ms"), float("nan"))
            slo = as_float(row.get("slo_tpot_ms"), float("nan"))
            if is_number(p99) and is_number(slo) and slo > 0 and p99 > 100 * slo:
                unit_ok = 0
                break
        out.append({
            "tag": exp["tag"],
            "kind": exp["kind"],
            "expected_cells": len(expected),
            "completed_cells": len({(r.get("tag"), int(r.get("N")), r.get("policy")) for r in selected if r.get("N") != ""}),
            "failed_cells": len([cell for cell in expected if cell in failed]),
            "missing_cells": json.dumps(missing, sort_keys=True),
            "goodput_all_zero": int(bool(goodputs) and max(goodputs) <= 0.0),
            "attainment_all_zero": int(bool(attain) and max(attain) <= 0.0),
            "max_kv_evictions": max(evictions) if evictions else 0.0,
            "kv_sweep_pressure_seen": int(max(evictions) > 0) if exp["kind"] == "kv_sweep" and evictions else -1,
            "max_verifier_load_rho": max(rhos) if rhos else 0.0,
            "rho_exceeds_1": int(bool(rhos) and max(rhos) > 1.0),
            "indep_opt_collapsed": int(bool(indep) and max(indep) <= 0.0) if any(p == "IndepOpt" for p in exp["policies"]) else -1,
            "unit_check_p99_lt_100x_slo": unit_ok,
        })
    return out


def filter_kind(rows: Iterable[dict], kind: str) -> List[dict]:
    return [r for r in rows if r.get("kind") == kind]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper3090.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    root = Path(cfg["paths"]["results_root"]).resolve()
    raw = root / "raw"
    processed = root / "processed"
    all_rows = collect_scheduler(raw, cfg)
    allowed_tag_n = configured_tag_n(cfg)
    run_status = read_csv(raw / "run_status.csv")
    request_rows = collect_named(raw, "scheduler_requests", cfg, allowed_tag_n)

    write_csv(processed / "scheduler_all_summary.csv", all_rows)
    write_csv(processed / "scheduler_main_summary.csv", filter_kind(all_rows, "main"))
    write_csv(processed / "scheduler_slo_sensitivity_summary.csv", filter_kind(all_rows, "slo_sensitivity"))
    write_csv(processed / "scheduler_kv_sweep_summary.csv", filter_kind(all_rows, "kv_sweep"))
    write_csv(processed / "ablation_summary.csv", filter_kind(all_rows, "ablation"))
    write_csv(processed / "scheduler_transfer_gpu_summary.csv", filter_kind(all_rows, "transfer_gpu"))
    write_csv(processed / "tier_fairness_summary.csv", tier_fairness_summary(all_rows, request_rows))
    write_csv(processed / "nospec_tpot_by_request.csv", collect_named(raw, "nospec_tpot_by_request", cfg, allowed_tag_n))
    write_csv(processed / "slo_thresholds.csv", collect_named(raw, "slo_thresholds", cfg, allowed_tag_n))
    write_csv(processed / "quality_checks.csv", quality_checks(all_rows, cfg, run_status))
    if run_status:
        write_csv(processed / "run_status.csv", run_status)
    summary = {
        "raw_dir": str(raw),
        "processed_dir": str(processed),
        "scheduler_rows": len(all_rows),
        "run_status_rows": len(run_status),
        "expected_cells": len(expected_cells(cfg)),
    }
    (processed / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote processed results to {processed}")


if __name__ == "__main__":
    main()

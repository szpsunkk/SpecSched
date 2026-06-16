#!/usr/bin/env python3
"""Print progress for a configured multi-seed validation run."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    return json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tail(path: Path, lines: int = 8) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def interesting_tail(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    interesting = [
        line
        for line in lines
        if line.startswith("[paper-config]")
        or line.startswith("[slo]")
        or line.startswith("[bench]")
        or line.startswith("[calib] IndepOpt")
        or "[ERROR]" in line
        or "Traceback" in line
        or "CUDA out of memory" in line
    ]
    return interesting[-1] if interesting else (lines[-1] if lines else "")


def phase_from_log(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if "[bench]" in text:
        return "running"
    if "[slo] pilot" in text:
        return "slo_pilot"
    if "[calib]" in text:
        return "calibrating"
    return "starting"


def pid_alive(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return "empty"
    try:
        pid = int(text.split()[0])
        os.kill(pid, 0)
        return f"alive:{pid}"
    except ProcessLookupError:
        return f"dead:{text}"
    except Exception:
        return text


def exit_code_text(exitfile: Path, logfile: Path, wrapper_state: str) -> str:
    if exitfile.exists():
        return exitfile.read_text(encoding="utf-8").strip()
    log_tail = tail(logfile, 80)
    for line in reversed(log_tail.splitlines()):
        if "exit_code=" in line:
            return line.split("exit_code=", 1)[1].split()[0]
    if wrapper_state.startswith("dead:"):
        return "unknown"
    return "running"


def scheduler_process_lines(results_root: Path | None = None) -> list[str]:
    pattern = "experiments/scheduler/run_scheduler.py"
    proc = subprocess.run(
        ["pgrep", "-f", pattern],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    pids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not pids:
        return []
    ps = subprocess.run(
        ["ps", "-o", "pid=,etime=,time=,%cpu=,%mem=,rss=,stat=,cmd=", "-p", ",".join(pids)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    lines = []
    for line in ps.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            lines.append(line)
            continue
        pid, etime, cputime, cpu, mem, rss, stat, cmd = parts
        if results_root is not None and str(results_root) not in cmd:
            continue
        args = cmd.split()
        n_value = ""
        policy_hint = ""
        if "--Ns" in args:
            idx = args.index("--Ns")
            if idx + 1 < len(args):
                n_value = args[idx + 1]
        if "--policies" in args:
            idx = args.index("--policies")
            if idx + 1 < len(args):
                policy_hint = args[idx + 1].split(",")[0]
        lines.append(
            f"pid={pid} etime={etime} cputime={cputime} cpu={cpu}% mem={mem}% "
            f"rss_kb={rss} stat={stat} N={n_value} first_policy={policy_hint}"
        )
    return lines


def expected_cells(cfg: dict) -> list[tuple[int, int]]:
    return [(int(seed), int(n)) for seed in cfg["seeds"] for n in cfg["Ns"]]


def scalar_or_by_n(value, n_value: int, default="") -> str:
    if isinstance(value, dict):
        return str(value.get(str(n_value), value.get(n_value, default)))
    if value is None:
        return str(default)
    return str(value)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    if not out.get("configured_kv_blocks") and out.get("x_kv_blocks"):
        out["configured_kv_blocks"] = out["x_kv_blocks"]
    if not out.get("configured_beta") and out.get("x_beta"):
        out["configured_beta"] = out["x_beta"]
    return out


def group_fields(cfg: dict) -> list[str]:
    return [str(field) for field in cfg.get("group_fields", [])]


def group_tuple(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def group_label(fields: list[str], values: tuple[str, ...]) -> str:
    if not fields:
        return ""
    return " ".join(f"{field}={value}" for field, value in zip(fields, values))


def expected_group_values(cfg: dict) -> list[tuple[str, ...]]:
    fields = group_fields(cfg)
    if not fields:
        return [tuple()]
    configured = cfg.get("expected_groups", [])
    if configured:
        return [tuple(str(item.get(field, "")) for field in fields) for item in configured]
    return [tuple("" for _ in fields)]


def row_matches_current(row: dict[str, str], base_cfg: dict, n_value: int) -> bool:
    common = base_cfg.get("common", {})
    arrival_model = str(common.get("arrival_model", "closed_loop"))
    requests_per_n = scalar_or_by_n(common.get("requests_per_N"), n_value, "")
    expected_request_count = ""
    if requests_per_n != "":
        try:
            requests_per_n_int = int(float(requests_per_n))
            if arrival_model == "closed_loop":
                expected_request_count = str(n_value)
            else:
                expected_request_count = str(max(n_value, requests_per_n_int))
        except ValueError:
            expected_request_count = requests_per_n
    want = {
        "N": str(n_value),
        "arrival_model": arrival_model,
        "arrival_rate_rps": scalar_or_by_n(common.get("arrival_rate_rps", 0.0), n_value, "0.0"),
        "x_max_batch": str(common.get("max_batch", "")),
        "x_block_tokens": str(common.get("block_tokens", "")),
        "x_kv_blocks": str(common.get("kv_blocks", "")),
        "x_kmax": str(common.get("kmax", "")),
    }
    request_actual = row.get("x_request_count", row.get("requests", ""))
    if expected_request_count != "" and request_actual != "":
        want["request_count"] = expected_request_count
    for key, expected in want.items():
        if expected == "":
            continue
        if key == "request_count":
            actual = str(request_actual)
        else:
            actual = str(row.get(key, ""))
        try:
            if abs(float(actual) - float(expected)) > 1e-9:
                return False
        except ValueError:
            if actual != expected:
                return False
    return True


def fmt_float(value: str, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "nan"


def policy_metric_summary(rows: list[dict[str, str]], policy_order: list[str]) -> str:
    if not rows:
        return ""
    order = {policy: idx for idx, policy in enumerate(policy_order)}
    rows = sorted(rows, key=lambda row: order.get(str(row.get("policy", "")), 10_000))
    latest = rows[-1]
    best = max(rows, key=lambda row: float(row.get("goodput_tok_s", "nan") or "nan"))
    latest_policy = str(latest.get("policy", ""))
    latest_summary = (
        f"latest={latest_policy}:gp={fmt_float(latest.get('goodput_tok_s', ''))}"
        f",p99={fmt_float(latest.get('p99_tpot_ms', ''), 1)}"
        f",att={fmt_float(latest.get('slo_attainment', ''), 3)}"
    )
    best_policy = str(best.get("policy", ""))
    best_summary = f"best={best_policy}:gp={fmt_float(best.get('goodput_tok_s', ''))}"
    return f" {latest_summary} {best_summary}"


def reference_rows_for_seed(cfg: dict, seed: int, n_value: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in cfg.get("reference_results", []):
        if int(item.get("seed", -1)) != seed:
            continue
        for row in read_csv(Path(item["processed_summary"])):
            if str(row.get("N", "")) == str(n_value):
                rows.append(row)
    return rows


def logs_for_group(raw: Path, n_value: int, fields: list[str], group_values: tuple[str, ...]) -> list[Path]:
    candidates = sorted((raw / "logs").glob(f"*N{n_value}.log"))
    if not fields:
        return candidates
    filters = {field: value for field, value in zip(fields, group_values)}
    kv_blocks = filters.get("configured_kv_blocks")
    if kv_blocks:
        kv_tag = f"kv{kv_blocks}"
        matched = [path for path in candidates if kv_tag in path.name]
        return matched
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tpds_multiseed_main.json")
    parser.add_argument("--tail-lines", type=int, default=10)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base_cfg = load_config(Path(cfg["base_config"]).resolve())
    results_root = Path(cfg["results_root"])
    pidfile = results_root / "multiseed_background.pid"
    exitfile = results_root / "multiseed_background.exit"
    logfile = results_root / "multiseed_background.log"
    policies = [str(policy) for policy in cfg.get("aggregate_policies", cfg["policies"])]

    print(f"config: {args.config}")
    print(f"results_root: {results_root}")
    wrapper_state = pid_alive(pidfile)
    print(f"wrapper: {wrapper_state}")
    print(f"exit_code: {exit_code_text(exitfile, logfile, wrapper_state)}")
    active_processes = scheduler_process_lines(results_root)
    if active_processes:
        print("scheduler_processes:")
        for line in active_processes:
            print(f"  {line}")
    print("")
    print("cells:")
    for item in cfg.get("reference_results", []):
        seed = int(item["seed"])
        for n_value in cfg["Ns"]:
            rows = reference_rows_for_seed(cfg, seed, int(n_value))
            completed = {str(row.get("policy", "")) for row in rows} & set(policies)
            metrics = policy_metric_summary(
                [row for row in rows if str(row.get("policy", "")) in policies],
                policies,
            )
            status = "ok" if len(completed) >= len(policies) else "incomplete_reference"
            print(
                f"  seed={seed} N={n_value}: status={status} "
                f"policies={len(completed)}/{len(policies)} source=reference{metrics}"
            )
    for seed, n_value in expected_cells(cfg):
        raw = results_root / f"seed_{seed}" / "raw"
        rows = read_csv(raw / "run_status.csv")
        by_n = [row for row in rows if str(row.get("N", "")) == str(n_value)]
        status = by_n[-1].get("status", "") if by_n else "not_started"
        elapsed = by_n[-1].get("elapsed_s", "") if by_n else ""
        completed_policies = set()
        completed_rows: list[dict[str, str]] = []
        for path in raw.glob(f"*/scheduler_*_N{n_value}_*.csv"):
            if path.name.startswith("scheduler_requests_") or path.name.startswith("scheduler_calib_"):
                continue
            for row in read_csv(path):
                row = normalize_row(row)
                if row_matches_current(row, base_cfg, n_value):
                    completed_policies.add(str(row.get("policy", "")))
                    completed_rows.append(row)
        fields = group_fields(cfg)
        groups = expected_group_values(cfg)
        if not fields:
            groups = [tuple()]
        for group_values in groups:
            if fields:
                group_rows = [row for row in completed_rows if group_tuple(row, fields) == group_values]
            else:
                group_rows = completed_rows
            completed_policies = {str(row.get("policy", "")) for row in group_rows}
            count = len(completed_policies & set(policies))
            active = ""
            metrics = policy_metric_summary(group_rows, policies)
            log_candidates = logs_for_group(raw, n_value, fields, group_values)
            cell_status = status
            cell_elapsed = elapsed
            if count >= len(policies):
                cell_status = "ok"
            elif count > 0 and log_candidates:
                cell_status = "running"
                cell_elapsed = ""
                log_path = log_candidates[-1]
                age_s = max(0.0, time.time() - log_path.stat().st_mtime)
                active = f" active_log={log_path.name} log_age_s={age_s:.0f} last='{interesting_tail(log_path)}'"
            elif count == 0 and log_candidates:
                cell_elapsed = ""
                log_path = log_candidates[-1]
                cell_status = phase_from_log(log_path)
                age_s = max(0.0, time.time() - log_path.stat().st_mtime)
                active = f" active_log={log_path.name} log_age_s={age_s:.0f} last='{interesting_tail(log_path)}'"
            elif count == 0:
                cell_elapsed = ""
            label = group_label(fields, group_values)
            label = f" {label}" if label else ""
            print(
                f"  seed={seed} N={n_value}{label}: status={cell_status} "
                f"policies={count}/{len(policies)} elapsed_s={cell_elapsed}{metrics}{active}"
            )

    print("")
    print("log tail:")
    print(tail(logfile, args.tail_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

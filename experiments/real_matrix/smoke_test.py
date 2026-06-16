"""Smoke test for the configured paper experiment framework.

This intentionally uses tiny parameters from configs/smoke_paper.json. It
checks process startup, raw/processed artifacts, and schema completeness only;
the produced performance values are not meaningful.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from .config_utils import load_config as load_config_file
except ImportError:
    from config_utils import load_config as load_config_file


ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FIELDS = {
    "policy",
    "N",
    "requests",
    "tokens",
    "wall_s",
    "throughput_tok_s",
    "goodput_tok_s",
    "slo_attainment",
    "p99_tpot_ms",
    "acceptance_rate",
    "target_forwards",
    "draft_forwards",
    "service_target_ms",
    "service_draft_ms",
    "service_other_ms",
    "service_target_frac",
    "service_draft_frac",
    "service_other_frac",
    "kv_evictions",
    "mean_selected_k",
    "verifier_load_rho",
    "x_max_batch",
    "x_kmax",
    "x_kv_blocks",
    "x_block_tokens",
    "x_seed",
    "x_deterministic",
    "x_peak_gpu_mb",
}
REQUEST_FIELDS = {
    "req_id",
    "bucket",
    "prompt_len",
    "generated",
    "target_forwards",
    "draft_forwards",
    "accepted",
    "drafted",
    "service_target_ms",
    "service_draft_ms",
    "service_other_ms",
    "ttft_ms",
    "tpot_ms",
    "policy",
    "N",
}
CALIB_FIELDS = {
    "calib_id",
    "prompt_len",
    "bucket",
    "k",
    "tokens",
    "service_s",
    "tokens_per_s",
    "accepted",
    "drafted",
    "acceptance",
}
SLO_FIELDS = {
    "request_id",
    "class",
    "N",
    "median_tpot_nospec",
    "median_ttft_nospec",
    "beta",
    "tau_slo",
    "tau_ttft",
    "unit",
}
NOSPEC_FIELDS = {
    "req_id",
    "bucket",
    "prompt_len",
    "generated",
    "ttft_ms",
    "tpot_ms",
    "pilot",
    "N",
}
RUN_STATUS_FIELDS = {
    "index",
    "total",
    "experiment",
    "kind",
    "tag",
    "N",
    "policies",
    "seed",
    "runtime_config_hash",
    "hyperparam_hash",
    "log",
    "command",
    "status",
    "exit_code",
}
PROCESSED_FILES = [
    "scheduler_all_summary.csv",
    "scheduler_main_summary.csv",
    "ablation_summary.csv",
    "nospec_tpot_by_request.csv",
    "slo_thresholds.csv",
    "quality_checks.csv",
    "run_status.csv",
    "processing_summary.json",
]


def run(cmd: List[str]) -> None:
    print("[smoke] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise AssertionError(f"missing expected CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def assert_fields(path: Path, required: Iterable[str]) -> List[dict]:
    rows = read_csv(path)
    if not rows:
        raise AssertionError(f"CSV has no data rows: {path}")
    missing = set(required) - set(rows[0])
    if missing:
        raise AssertionError(f"{path} missing fields: {sorted(missing)}")
    return rows


def expected_cells(cfg: dict) -> List[dict]:
    cells = []
    for exp in cfg["experiments"]:
        for n_value in exp["Ns"]:
            cells.append({"exp": exp, "N": int(n_value)})
    return cells


def expected_policies(cfg: dict) -> set:
    out = set()
    for exp in cfg["experiments"]:
        for n_value in exp["Ns"]:
            for policy in exp["policies"]:
                out.add((exp["tag"], int(n_value), policy))
    return out


def find_one(paths: Iterable[Path], pattern: str) -> Path:
    matches = sorted(path for root in paths for path in root.glob(pattern))
    if not matches:
        raise AssertionError(f"missing file matching {pattern}")
    return matches[0]


def validate_raw(cfg: dict, raw: Path) -> None:
    manifest = raw / "run_manifest.json"
    effective = raw / "effective_config.json"
    status_jsonl = raw / "run_status.jsonl"
    for path in (manifest, effective, status_jsonl):
        if not path.exists():
            raise AssertionError(f"missing expected JSON artifact: {path}")
    json.loads(manifest.read_text(encoding="utf-8"))
    json.loads(effective.read_text(encoding="utf-8"))
    status_rows = assert_fields(raw / "run_status.csv", RUN_STATUS_FIELDS)
    if len(status_rows) != len(expected_cells(cfg)):
        raise AssertionError(f"run_status row count {len(status_rows)} != expected {len(expected_cells(cfg))}")
    failed = [row for row in status_rows if row.get("status") != "ok" or row.get("exit_code") != "0"]
    if failed:
        raise AssertionError(f"smoke cells did not finish cleanly: {failed}")

    for cell in expected_cells(cfg):
        exp = cell["exp"]
        n_value = cell["N"]
        tag_dir = raw / exp["tag"]
        suffix = f"{exp['tag']}_N{n_value}_*.csv"
        assert_fields(find_one([tag_dir], f"scheduler_{suffix}"), SUMMARY_FIELDS)
        assert_fields(find_one([tag_dir], f"scheduler_requests_{suffix}"), REQUEST_FIELDS)
        assert_fields(find_one([tag_dir], f"scheduler_calib_{suffix}"), CALIB_FIELDS)
        assert_fields(find_one([tag_dir], f"slo_thresholds_{suffix}"), SLO_FIELDS)
        assert_fields(find_one([tag_dir], f"nospec_tpot_by_request_{suffix}"), NOSPEC_FIELDS)


def validate_processed(cfg: dict, root: Path) -> None:
    processed = root / "processed"
    for name in PROCESSED_FILES:
        path = processed / name
        if not path.exists():
            raise AssertionError(f"missing processed artifact: {path}")
    summary = json.loads((processed / "processing_summary.json").read_text(encoding="utf-8"))
    if int(summary.get("expected_cells", -1)) != len(expected_policies(cfg)):
        raise AssertionError(f"unexpected expected_cells in processing summary: {summary}")

    rows = assert_fields(processed / "scheduler_all_summary.csv", SUMMARY_FIELDS | {"tag", "experiment", "kind"})
    observed = {(row["tag"], int(float(row["N"])), row["policy"]) for row in rows}
    missing = expected_policies(cfg) - observed
    if missing:
        raise AssertionError(f"processed scheduler summary missing cells: {sorted(missing)}")

    kinds = {str(exp.get("kind", "")) for exp in cfg["experiments"]}
    if "main" in kinds:
        assert_fields(processed / "scheduler_main_summary.csv", SUMMARY_FIELDS | {"tag", "kind"})
    if "ablation" in kinds:
        assert_fields(processed / "ablation_summary.csv", SUMMARY_FIELDS | {"tag", "kind"})
    if "kv_sweep" in kinds:
        assert_fields(processed / "scheduler_kv_sweep_summary.csv", SUMMARY_FIELDS | {"tag", "kind"})
    if "slo_sensitivity" in kinds:
        assert_fields(processed / "scheduler_slo_sensitivity_summary.csv", SUMMARY_FIELDS | {"tag", "kind"})
    assert_fields(processed / "nospec_tpot_by_request.csv", NOSPEC_FIELDS | {"tag", "kind"})
    assert_fields(processed / "slo_thresholds.csv", SLO_FIELDS | {"tag", "kind"})
    assert_fields(processed / "quality_checks.csv", {"tag", "kind", "expected_cells", "completed_cells", "missing_cells"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_paper.json")
    parser.add_argument("--keep-results", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    loaded = load_config_file(config_path, ROOT)
    cfg = loaded.cfg
    root = Path(cfg["paths"]["results_root"]).resolve()
    if root.exists() and not args.keep_results:
        shutil.rmtree(root)

    run([sys.executable, "experiments/real_matrix/run_configured_paper.py", "--config", str(config_path)])
    run([sys.executable, "experiments/real_matrix/process_configured_results.py", "--config", str(config_path)])
    validate_raw(cfg, root / "raw")
    validate_processed(cfg, root)
    print(f"[smoke] passed; results_root={root}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write a compact internal progress report for the TPDS revision work."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def md_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ")


def fmt_pct(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return md_cell(value)


def gap_context(row: dict) -> str:
    bits = []
    for key in ("result_set", "N", "kv_blocks"):
        value = row.get(key)
        if value not in (None, ""):
            bits.append(f"{key}={value}")
    return ", ".join(bits)


def gap_metric(row: dict) -> str:
    if row.get("ci95_low_pct") not in (None, "") or row.get("ci95_high_pct") not in (None, ""):
        return (
            f"lift {fmt_pct(row.get('mean_lift_pct'))} "
            f"CI [{fmt_pct(row.get('ci95_low_pct'))}, {fmt_pct(row.get('ci95_high_pct'))}]"
        )
    if row.get("wins") not in (None, "") or row.get("total") not in (None, ""):
        return f"wins {row.get('wins', '')}/{row.get('total', '')}"
    if row.get("evictv2_lift_vs_lru_pct") not in (None, ""):
        return (
            f"EvictV2 {row.get('evictv2_goodput', '')}, "
            f"LRU {row.get('lru_goodput', '')}, "
            f"lift {fmt_pct(row.get('evictv2_lift_vs_lru_pct'))}"
        )
    if row.get("observed_model_pairs") not in (None, ""):
        return f"model_pairs={row.get('observed_model_pairs')}"
    if row.get("max_degradation_pct") not in (None, ""):
        return f"offline_rank_flip_proxy={fmt_pct(row.get('max_degradation_pct'))}"
    if row.get("max_abs_goodput_delta_pct") not in (None, ""):
        return f"max_abs_delta={fmt_pct(row.get('max_abs_goodput_delta_pct'))}"
    if row.get("mae_ms") not in (None, "") or row.get("r2") not in (None, ""):
        return f"MAE={row.get('mae_ms', '')}ms, R2={row.get('r2', '')}"
    return ""


def open_gap_rows(gap_data: dict) -> list[str]:
    failed = [
        row for row in gap_data.get("checks", [])
        if str(row.get("pass", "")).lower() in ("0", "false", "no")
    ]
    if not failed:
        return [
            "## Open Gap Details",
            "",
            "No open gaps reported by `scripts/audit_tpds_gaps.py`.",
        ]

    rows = [
        "## Open Gap Details",
        "",
        "| gap | check | context | metric | evidence / note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in failed:
        evidence = row.get("evidence") or row.get("note") or ""
        rows.append(
            f"| {md_cell(row.get('gap', ''))} | "
            f"{md_cell(row.get('check', ''))} | "
            f"{md_cell(gap_context(row))} | "
            f"{md_cell(gap_metric(row))} | "
            f"{md_cell(evidence)} |"
        )
    return rows


def command_stdout(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout.strip()


def final_poisson_rows() -> list[str]:
    paired = read_csv(ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_paired_lifts.csv")
    summary = read_csv(ROOT / "results/tpds_multiseed_heterog_poisson/processed/main_multiseed_summary.csv")
    if not paired or not summary:
        return []

    specsched_by_n = {
        row["N"]: row
        for row in summary
        if row.get("policy") == "SpecSched"
    }
    bestfixed = {
        row["N"]: row
        for row in paired
        if row.get("baseline") == "BestFixed"
    }
    rows = [
        "## Heterogeneous Poisson Final Multi-Seed Summary",
        "",
        "| N | SpecSched goodput | vs NoSpec | vs TETRIS-Proxy | vs BestFixed | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for n_value in ("16", "32", "64"):
        spec = specsched_by_n.get(n_value, {})
        no_spec = next((row for row in paired if row.get("N") == n_value and row.get("baseline") == "NoSpec"), {})
        tetris = next((row for row in paired if row.get("N") == n_value and row.get("baseline") == "TETRIS-Proxy"), {})
        fixed = bestfixed.get(n_value, {})
        note = "BestFixed CI crosses zero; do not claim a fixed-k win"
        rows.append(
            f"| {n_value} | "
            f"{spec.get('goodput_tok_s_mean', '')} +/- {spec.get('goodput_tok_s_std', '')} | "
            f"{no_spec.get('goodput_lift_pct_mean', '')}% "
            f"[{no_spec.get('goodput_lift_pct_ci95_low', '')}, {no_spec.get('goodput_lift_pct_ci95_high', '')}] | "
            f"{tetris.get('goodput_lift_pct_mean', '')}% "
            f"[{tetris.get('goodput_lift_pct_ci95_low', '')}, {tetris.get('goodput_lift_pct_ci95_high', '')}] | "
            f"{fixed.get('goodput_lift_pct_mean', '')}% "
            f"[{fixed.get('goodput_lift_pct_ci95_low', '')}, {fixed.get('goodput_lift_pct_ci95_high', '')}] | "
            f"{note} |"
        )
    return rows


def guarded_targeted_rows() -> list[str]:
    rows = read_csv(ROOT / "results/evict_guarded_targeted/processed/scheduler_kv_sweep_summary.csv")
    if not rows:
        return []

    by_cell: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        budget = row.get("configured_kv_blocks") or row.get("x_kv_blocks") or ""
        n_value = row.get("N", "")
        if budget and n_value:
            by_cell.setdefault((budget, n_value), []).append(row)

    lines = [
        "## EvictGuarded Targeted Sweep",
        "",
        "This targeted 7B sweep is internal evidence for A3 only. It does not support adding a guarded-eviction claim to the manuscript.",
        "",
        "| KV blocks | N | best policy | best goodput | Guarded goodput | Guarded vs LRU | Guarded vs EvictV2 | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    guarded_wins = 0
    total = 0
    for key in sorted(by_cell, key=lambda x: (int(float(x[0])), int(float(x[1])))):
        cell_rows = by_cell[key]
        best = max(cell_rows, key=lambda r: float(r.get("goodput_tok_s", "0") or 0))
        guarded = next((r for r in cell_rows if r.get("policy") == "SpecSched-EvictGuarded"), {})
        lru = next((r for r in cell_rows if r.get("policy") == "SpecSched-LRU"), {})
        evictv2 = next((r for r in cell_rows if r.get("policy") == "SpecSched-EvictV2"), {})
        guarded_gp = float(guarded.get("goodput_tok_s", "0") or 0)
        lru_gp = float(lru.get("goodput_tok_s", "0") or 0)
        evictv2_gp = float(evictv2.get("goodput_tok_s", "0") or 0)
        best_gp = float(best.get("goodput_tok_s", "0") or 0)
        total += 1
        guarded_wins += int(guarded.get("policy") == best.get("policy"))
        note = "Guarded not best; keep manuscript conservative"
        if guarded.get("policy") == best.get("policy"):
            note = "Guarded best on this cell only"
        lines.append(
            f"| {key[0]} | {key[1]} | "
            f"{best.get('policy', '')} | {best_gp:.2f} | {guarded_gp:.2f} | "
            f"{((guarded_gp / lru_gp - 1.0) * 100.0):.2f}% | "
            f"{((guarded_gp / evictv2_gp - 1.0) * 100.0):.2f}% | "
            f"{note} |"
        )
    lines.extend([
        "",
        f"Guarded is best in {guarded_wins}/{total} targeted cells. This confirms that A3 should remain framed as an operating-regime characterization, not as a universal win over LRU or vLLM-style eviction.",
    ])
    return lines


def second_model_rows() -> list[str]:
    summary = read_csv(ROOT / "results/tpds_multiseed_second_model/processed/main_multiseed_summary.csv")
    paired = read_csv(ROOT / "results/tpds_multiseed_second_model/processed/main_multiseed_paired_lifts.csv")
    if not summary:
        return []
    by_policy = {row.get("policy", ""): row for row in summary}
    spec = by_policy.get("SpecSched", {})
    nospec = by_policy.get("NoSpec", {})
    fixed1 = by_policy.get("FixedK-1", {})

    def gp(row: dict[str, str]) -> float:
        return float(row.get("goodput_tok_s_mean", "0") or 0)

    def gp_std(row: dict[str, str]) -> float:
        return float(row.get("goodput_tok_s_std", "0") or 0)

    def lift_row(baseline: str) -> dict[str, str]:
        return next((row for row in paired if row.get("baseline") == baseline), {})

    nospec_lift = lift_row("NoSpec")
    fixed_lift = lift_row("BestFixed")
    tetris_lift = lift_row("TETRIS-Proxy")
    lines = [
        "## Second-Model Multi-Seed Boundary Check",
        "",
        "This is a model-pair boundary check, not a universal performance claim.",
        "",
        f"- N: {spec.get('N', 'missing')}, seeds: {spec.get('seeds', 'missing')}",
        f"- SpecSched goodput: {gp(spec):.2f} +/- {gp_std(spec):.2f} tok/s",
        f"- NoSpec goodput: {gp(nospec):.2f} +/- {gp_std(nospec):.2f} tok/s",
        f"- FixedK-1 goodput: {gp(fixed1):.2f} +/- {gp_std(fixed1):.2f} tok/s",
        f"- SpecSched vs NoSpec: {nospec_lift.get('goodput_lift_pct_mean', 'missing')}% "
        f"[{nospec_lift.get('goodput_lift_pct_ci95_low', '')}, {nospec_lift.get('goodput_lift_pct_ci95_high', '')}]",
        f"- SpecSched vs TETRIS-Proxy: {tetris_lift.get('goodput_lift_pct_mean', 'missing')}% "
        f"[{tetris_lift.get('goodput_lift_pct_ci95_low', '')}, {tetris_lift.get('goodput_lift_pct_ci95_high', '')}]",
        f"- SpecSched vs BestFixed: {fixed_lift.get('goodput_lift_pct_mean', 'missing')}% "
        f"[{fixed_lift.get('goodput_lift_pct_ci95_low', '')}, {fixed_lift.get('goodput_lift_pct_ci95_high', '')}]",
        "",
        "SpecSched is below NoSpec and BestFixed on this smaller 1.5B target, where draft and verification overheads dominate. The manuscript should use this as a boundary condition.",
    ]
    return lines


def normalize_status(status: str) -> str:
    if "exit_code: running" not in status:
        return status
    replacement = None
    for line in status.splitlines():
        if line.startswith("[background]") and "exit_code=" in line:
            replacement = line.split("exit_code=", 1)[1].split()[0]
    if replacement is None:
        return status
    return status.replace("exit_code: running", f"exit_code: {replacement}", 1)


def main() -> int:
    gap_path = ROOT / "results/tpds_gap_audit_current/tpds_gap_audit.json"
    if not gap_path.exists():
        gap_path = ROOT / "results/tpds_gap_audit/tpds_gap_audit.json"
    gap_data = read_json(gap_path)
    gap = gap_data.get("summary", {})
    perturb = read_json(ROOT / "results/verify_latency_fit/processed/verify_latency_perturbation.json")
    posalpha = read_json(ROOT / "results/estimator_audit/processed/posalpha_estimator_audit.json")
    partial = read_csv(ROOT / "results/tpds_multiseed_heterog_poisson/processed_partial/partial_diagnostics.csv")
    readiness_tail = command_stdout(["python", "scripts/check_tpds_readiness.py"]).splitlines()[-8:]
    heterog_status = normalize_status(command_stdout([
        "python",
        "experiments/real_matrix/status_multiseed.py",
        "--config",
        "configs/tpds_multiseed_heterog_poisson.json",
    ]))
    kv_status = normalize_status(command_stdout([
        "python",
        "experiments/real_matrix/status_multiseed.py",
        "--config",
        "configs/tpds_multiseed_kv_pressure.json",
    ]))

    lines = [
        "# TPDS Revision Progress Report",
        "",
        "This is an internal progress report generated from local audit outputs.",
        "Do not upload it as a manuscript artifact.",
        "",
        "## Readiness",
        "",
        "```text",
        *readiness_tail,
        "```",
        "",
        "## Gap Audit",
        "",
        f"- checks: {gap.get('checks', 'unknown')}",
        f"- open_gaps: {gap.get('open_gaps', 'unknown')}",
        f"- status: {gap.get('status', 'unknown')}",
        "",
        "## Focused Audits",
        "",
        f"- Verify-latency perturbation max offline rank-flip proxy: {perturb.get('max_degradation_pct', 'missing')}%",
        f"- PosAlpha max absolute goodput delta: {posalpha.get('max_abs_goodput_delta_pct', 'missing')}%",
        "",
    ]
    lines.extend(open_gap_rows(gap_data))
    lines.append("")

    final_rows = final_poisson_rows()
    if final_rows:
        lines.extend(final_rows)
    else:
        lines.extend([
            "## Heterogeneous Poisson Partial Diagnostics",
            "",
            "| seed | N | completed | best fixed | SpecSched | SpecSched vs BestFixed | note |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in partial:
            best = row.get("best_fixed_policy", "")
            best_gp = row.get("best_fixed_goodput_tok_s", "")
            spec = row.get("specsched_goodput_tok_s", "")
            lift = row.get("specsched_vs_bestfixed_pct", "")
            lines.append(
                f"| {row.get('seed','')} | {row.get('N','')} | "
                f"{row.get('completed_count','')}/{row.get('expected_count','')} | "
                f"{best} {best_gp} | {spec} | {lift} | {row.get('diagnostic_note','')} |"
            )
    guarded_rows = guarded_targeted_rows()
    if guarded_rows:
        lines.extend(["", *guarded_rows])
    model_rows = second_model_rows()
    if model_rows:
        lines.extend(["", *model_rows])
    lines.extend([
        "",
        "## Background Status: Heterogeneous Poisson",
        "",
        "```text",
        heterog_status,
        "```",
        "",
        "## Background Status: KV Pressure",
        "",
        "```text",
        kv_status,
        "```",
        "",
    ])

    out = ROOT / "main/tpds_progress_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

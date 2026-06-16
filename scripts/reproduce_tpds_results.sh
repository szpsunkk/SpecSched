#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/6] Rebuilding route/KV stress simulation from saved measured summaries"
python experiments/real_matrix/run_route_kv_stress_sim.py --config configs/route_kv_stress_sim.json

echo "[1b/6] Rebuilding BestFixed transfer stress simulation from saved measured summaries"
python experiments/real_matrix/run_bestfixed_stress_sim.py --config configs/bestfixed_stress_sim.json

echo "[2/6] Rebuilding main paper figures and tables from saved CSVs"
python scripts/plot_results.py --results-root results/local_v3_kcap3

echo "[3/6] Rebuilding edge deployment figures from saved CSVs"
python scripts/plot_edge_results.py --results-root results/tpds_multiseed_edge_practical

echo "[4/6] Rebuilding scientific-style figures"
python scripts/plot_scientific_figures.py

echo "[5/6] Rebuilding scheduler-overhead table from saved CSVs"
python scripts/generate_scheduler_overhead_table.py

echo "[6/6] Auditing headline manuscript claims against processed results"
python scripts/audit_paper_claims.py

echo "[post] Checking result-level artifact completeness"
python scripts/verify_tpds_artifact.py

echo "Result-level TPDS reproduction completed. No GPU experiments were rerun."

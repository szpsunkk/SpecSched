#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

LOG="results/tpds_multiseed_heterog_poisson/multiseed_background.log"
mkdir -p "$(dirname "$LOG")"

echo "[background] tpds multiseed heterog poisson start $(date)"
echo "[background] python=$(command -v python)"
python --version
python experiments/real_matrix/run_multiseed_validation.py \
  --config configs/tpds_multiseed_heterog_poisson.json
code=$?
echo "[background] tpds multiseed heterog poisson exit_code=${code} $(date)"
exit "$code"

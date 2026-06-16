#!/usr/bin/env bash
# Config-driven 3090Ti paper sweep. All experiment parameters live in configs/.
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/paper3090.json}"

python experiments/real_matrix/run_configured_paper.py --config "${CONFIG}"
python experiments/real_matrix/process_configured_results.py --config "${CONFIG}"
python experiments/real_matrix/plot_configured_paper.py --config "${CONFIG}"

echo "[paper] complete"

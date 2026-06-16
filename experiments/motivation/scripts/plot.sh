#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python experiments/motivation/plot_serving.py \
  --outdir experiments/motivation/figures

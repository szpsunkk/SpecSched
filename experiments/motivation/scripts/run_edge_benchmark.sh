#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python experiments/motivation/serve_benchmark.py \
  --edge-hosts "${JETSON_HOSTS:-100.66.37.19,100.107.23.95,100.73.217.47}" \
  --edge-user "${JETSON_USER:-skk}" \
  --edge-password "${JETSON_PASSWORD:-111}" \
  --draft-remote "${DRAFT_REMOTE:-~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct}" \
  --Ns 4,8,16,32 \
  --requests-per-N 32 \
  --prompt-len 128 \
  --max-new-tokens 8 \
  --modes single,multi \
  --draft-backend edge \
  --out experiments/motivation/results/serving_benchmark.csv \
  --request-out experiments/motivation/results/serving_requests.csv

#!/usr/bin/env bash
# Compact status view for the real paper runs.
set -euo pipefail

cd "$(dirname "$0")/../../.."

echo "== Processes =="
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd \
  | grep -E 'run_configured|run_scheduler|local_v2_clean|edge_n64|wait_then_start|run_config_background|run_paper' \
  | grep -v grep || true

echo
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

echo
echo "== Edge N64 =="
if [[ -f results/edge_n64_specsched_fix/raw/logs/edge_n64.exit ]]; then
  echo "exit=$(cat results/edge_n64_specsched_fix/raw/logs/edge_n64.exit)"
else
  echo "exit=pending"
fi
tail -n 20 results/edge_n64_specsched_fix/raw/logs/paper3090_edge_n64_specsched_fix_N64.log 2>/dev/null || true

echo
echo "== Clean Local Watcher =="
if [[ -f results/local_v2_clean/raw/logs/wait_then_start.pid ]]; then
  pid="$(cat results/local_v2_clean/raw/logs/wait_then_start.pid)"
  ps -p "$pid" -o pid,ppid,sid,stat,etime,%cpu,%mem,cmd || true
fi
tail -n 20 results/local_v2_clean/raw/logs/wait_then_start.log 2>/dev/null || true

echo
echo "== Clean Local Full Run =="
if [[ -f results/local_v2_clean/raw/logs/config_background.exit ]]; then
  echo "exit=$(cat results/local_v2_clean/raw/logs/config_background.exit)"
else
  echo "exit=pending/not-started"
fi
tail -n 20 results/local_v2_clean/raw/logs/config_background.log 2>/dev/null || true
tail -n 10 results/local_v2_clean/raw/run_status.csv 2>/dev/null || true

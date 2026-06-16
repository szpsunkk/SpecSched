#!/usr/bin/env bash
# Background wrapper for the paper sweep (run_paper.sh). Long-running; expect
# 6-12 hours on a 3090Ti depending on NS / KV_SWEEP / OUTPUT_LENGTHS.
set -euo pipefail

cd "$(dirname "$0")/../../.."
mkdir -p experiments/real_matrix/logs

LOG="results/raw/logs/paper_background.log"
PIDFILE="results/raw/logs/paper_background.pid"
EXITFILE="results/raw/logs/paper_background.exit"

mkdir -p results/raw/logs
rm -f "$EXITFILE"
setsid nohup bash -c '
  set +e
  echo "[background] paper start $(date)"
  bash experiments/real_matrix/scripts/run_paper.sh
  rc=$?
  echo "$rc" > results/raw/logs/paper_background.exit
  echo "[background] paper exit_code=${rc} $(date)"
  exit "$rc"
' > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PIDFILE"

echo "started paper experiment"
echo "log: $LOG"
echo "pid: $(cat "$PIDFILE")"

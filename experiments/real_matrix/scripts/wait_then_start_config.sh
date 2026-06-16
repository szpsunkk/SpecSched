#!/usr/bin/env bash
# Wait for one PID to exit, then start a config-driven experiment in background.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <wait_pid> <config>" >&2
  exit 2
fi

WAIT_PID="$1"
CONFIG_TO_START="$2"

cd "$(dirname "$0")/../../.."

echo "[watcher] waiting for PID ${WAIT_PID} at $(date)"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  echo "[watcher] still waiting for PID ${WAIT_PID} at $(date)"
  sleep 60
done

echo "[watcher] PID ${WAIT_PID} finished at $(date)"
if ps -eo cmd | grep -E "python experiments/scheduler/run_scheduler.py" | grep -v grep >/dev/null; then
  echo "[watcher] another scheduler is still running; not starting ${CONFIG_TO_START}"
  exit 2
fi

echo "[watcher] starting ${CONFIG_TO_START} at $(date)"
CONFIG="${CONFIG_TO_START}" bash experiments/real_matrix/scripts/run_config_background.sh

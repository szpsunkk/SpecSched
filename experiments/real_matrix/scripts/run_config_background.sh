#!/usr/bin/env bash
# Background wrapper for any config-driven paper sweep.
set -euo pipefail

cd "$(dirname "$0")/../../.."
CONFIG="${1:-${CONFIG:-configs/paper3090.json}}"
export CONFIG
RESULTS_ROOT="$(
  python - "$CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path.cwd()
os.environ.setdefault("PROJECT_ROOT", str(root))
text = Path(sys.argv[1]).read_text(encoding="utf-8")
cfg = json.loads(os.path.expandvars(text))
print(cfg["paths"]["results_root"])
PY
)"

mkdir -p "${RESULTS_ROOT}/raw/logs"
LOG="${RESULTS_ROOT}/raw/logs/config_background.log"
PIDFILE="${RESULTS_ROOT}/raw/logs/config_background.pid"
EXITFILE="${RESULTS_ROOT}/raw/logs/config_background.exit"

rm -f "$EXITFILE"
setsid nohup bash -c '
  set +e
  echo "[background] config=${CONFIG} start $(date)"
  bash experiments/real_matrix/scripts/run_paper.sh
  rc=$?
  echo "$rc" > "'"$EXITFILE"'"
  echo "[background] config=${CONFIG} exit_code=${rc} $(date)"
  exit "$rc"
' > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PIDFILE"

echo "started config experiment"
echo "config: $CONFIG"
echo "log: $LOG"
echo "pid: $(cat "$PIDFILE")"

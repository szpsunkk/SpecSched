#!/usr/bin/env bash
# Background wrapper for the focused TPDS multi-seed validation.
set -euo pipefail

cd "$(dirname "$0")/../../.."

CONFIG="${CONFIG:-configs/tpds_multiseed_main.json}"
ROOT="$(python - "$CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path

config = Path(sys.argv[1])
os.environ.setdefault("PROJECT_ROOT", str(Path.cwd()))
cfg = json.loads(os.path.expandvars(config.read_text(encoding="utf-8")))
print(cfg["results_root"])
PY
)"
LOG="${ROOT}/multiseed_background.log"
PIDFILE="${ROOT}/multiseed_background.pid"
EXITFILE="${ROOT}/multiseed_background.exit"

mkdir -p "$ROOT"
rm -f "$EXITFILE"

setsid nohup bash -c '
  set +e
  echo "[background] tpds multiseed start $(date)"
  python experiments/real_matrix/run_multiseed_validation.py --config "'"${CONFIG}"'"
  rc=$?
  echo "$rc" > "'"${EXITFILE}"'"
  echo "[background] tpds multiseed exit_code=${rc} $(date)"
  exit "$rc"
' > "$LOG" 2>&1 < /dev/null &

echo "$!" > "$PIDFILE"
echo "started TPDS multi-seed validation"
echo "log: $LOG"
echo "pid: $(cat "$PIDFILE")"

#!/usr/bin/env bash
# Train A then B back to back (each ~3.5-4 h on the A40 incl. sampling), then build A-vs-B progress grids
# from the checkpoint-time samples (CPU only). Training output goes to the screen and logs/train_<v>.log.
# Usually started through tmux_start.sh (survives ssh disconnects).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/logs"
for v in A B; do
  echo "[$(date '+%F %T')] train $v -> logs/train_$v.log"
  if ! bash "$HERE/train.sh" "$v" 2>&1 | tee "$HERE/logs/train_$v.log"; then
    echo "[$(date '+%F %T')] train $v FAILED (see logs/train_$v.log)"
  fi
done
echo "[$(date '+%F %T')] compare samples"
/root/venvs/sn15/bin/python "$HERE/compare_samples.py" || echo "[$(date '+%F %T')] compare FAILED"
echo "[$(date '+%F %T')] done"

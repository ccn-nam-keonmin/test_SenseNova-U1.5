#!/usr/bin/env bash
# ~10 min sanity run on 20 images: loading, VRAM, speed, checkpoint writing and checkpoint-time sampling
# (samples at step-0, step-10, step-20). Usage: smoke.sh [A|B]  (default A)
# Output goes to DiffSynth-Studio/models/train/sn15_smoke_<variant>.
set -euo pipefail
VARIANT=${1:-A}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/logs"
METADATA="$HERE/meta/smoke.csv" \
OUTPUT=/root/source/test_SenseNova-U1.5/DiffSynth-Studio/models/train/sn15_smoke_$VARIANT \
  bash "$HERE/train.sh" "$VARIANT" --num_epochs 1 --save_steps 10 2>&1 | tee "$HERE/logs/smoke_$VARIANT.log"

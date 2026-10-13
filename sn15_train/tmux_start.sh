#!/usr/bin/env bash
# Start a job in a detached tmux session so it keeps running after the ssh connection drops.
# Usage: tmux_start.sh [job]
#   smoke   smoke.sh A, then smoke.sh B (~20 min)
#   ab      run_ab.sh: train A, train B, compare samples (~7.5 h)   [default]
#   A | B   train.sh A or B only
#   test    prints a few lines (checks the tmux setup, no GPU)
# Window 0 "train" shows the job, window 1 "gpu" runs nvidia-smi every 5 s.
# Attach: tmux attach -t sn15   Detach (job keeps running): Ctrl-b then d
set -euo pipefail
JOB=${1:-ab}
SESSION=${SESSION:-sn15}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/logs"

case $JOB in
  smoke) CMD="bash $HERE/smoke.sh A && bash $HERE/smoke.sh B" ;;
  ab)    CMD="bash $HERE/run_ab.sh 2>&1 | tee -a $HERE/logs/run_ab.log" ;;
  A|B)   CMD="bash $HERE/train.sh $JOB 2>&1 | tee $HERE/logs/train_$JOB.log" ;;
  test)  CMD="echo tmux ok; nvidia-smi --query-gpu=name,memory.used --format=csv; sleep 3" ;;
  *) echo "unknown job '$JOB' (smoke | ab | A | B)" >&2; exit 1 ;;
esac

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION" >&2
  echo "(or kill it first: tmux kill-session -t $SESSION)" >&2
  exit 1
fi
if pgrep -f "sn15_train/train_sn15.py" >/dev/null; then
  echo "a training process is already running:" >&2
  pgrep -af "sn15_train/train_sn15.py" >&2
  exit 1
fi

# The job runs from a small runner script (no nested quoting); the pane stays open afterwards
# so the last output remains readable.
RUNNER="$HERE/logs/.tmux_$SESSION.sh"
cat > "$RUNNER" <<RUN
#!/usr/bin/env bash
set -o pipefail
echo "[\$(date '+%F %T')] job: $JOB"
$CMD
rc=\$?
echo
echo "[\$(date '+%F %T')] job finished, exit code \$rc"
exec bash
RUN
tmux new-session -d -s "$SESSION" -n train -c "$HERE" "bash $RUNNER"
tmux new-window -d -t "$SESSION" -n gpu "watch -n 5 nvidia-smi"
echo "started job '$JOB' in tmux session '$SESSION'"
echo "  watch:   tmux attach -t $SESSION      (detach: Ctrl-b then d; switch window: Ctrl-b then 0 / 1)"
echo "  logs:    $HERE/logs/"
echo "  stop:    tmux kill-session -t $SESSION"

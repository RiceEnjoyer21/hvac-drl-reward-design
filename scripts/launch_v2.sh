#!/usr/bin/env bash
set -u
cd ~/hvac-rl
source .venv/bin/activate
N="${1:-6}"
JOBS="${2:-configs/jobs_v2.txt}"
mkdir -p results/logs_v2
run_one() {
  local cfg="$1" seed="$2" name="${1}_s${2}"
  if [ -f "results/runs_v2/$name/model_final.zip" ]; then echo "skip $name (done)"; return; fi
  echo "start $name $(date +%H:%M)"
  python -u scripts/train_v2.py "configs/$cfg.yaml" --seed "$seed" 2>&1 \
    | grep --line-buffered -v -E '^#|INFO' > "results/logs_v2/$name.log"
  echo "end   $name $(date +%H:%M) $(tail -n 1 results/logs_v2/$name.log)"
}
export -f run_one
grep -v -E '^\s*(#|$)' "$JOBS" | xargs -P "$N" -L 1 bash -c 'run_one "$0" "$1"'
echo "all jobs finished $(date +%H:%M)"

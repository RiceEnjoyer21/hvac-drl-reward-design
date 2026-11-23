cd ~/hvac-rl
source .venv/bin/activate
mkdir -p results/logs_v2
for c in sac_r2_k50 td3_r2_k50 ppo_r2_k50 sac_res_k50 td3_res_k50 sac_r2_k50_base; do
  python -u scripts/train_v2.py configs/$c.yaml --seed 1 --steps 4096 --eval-freq 2048 --eval-steps 4000 \
    --name _smoke/$c > results/logs_v2/_smoke_$c.log 2>&1 &
done
wait
for c in sac_r2_k50 td3_r2_k50 ppo_r2_k50 sac_res_k50 td3_res_k50 sac_r2_k50_base; do
  echo "== $c"; grep -v -E '^#|INFO' results/logs_v2/_smoke_$c.log | tail -n 6
  ls results/runs_v2/_smoke/$c | tr '\n' ' '; echo
done

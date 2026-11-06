#!/usr/bin/env bash
# Execute the actual training curriculum from zero policy output on physical GPU 1.
set -euo pipefail
cd "$(dirname "$0")/.."
run_prefix="${1:-runs/inverted-pendulum}"
common=(--seed 42 --envs 64 --optimizer lbfgs --eval-every 1 --validation-steps 1000)

bash scripts/run_cuda.sh train_inverted_pendulum.py "${common[@]}" \
  --iterations 4 --horizon 64 --lr 1.0 --lbfgs-max-iter 20 \
  --out "${run_prefix}-h64"
bash scripts/run_cuda.sh train_inverted_pendulum.py "${common[@]}" \
  --iterations 3 --horizon 128 --lr 0.1 --lbfgs-max-iter 20 \
  --initial-policy "${run_prefix}-h64/latest.pt" --out "${run_prefix}-h128"
bash scripts/run_cuda.sh evaluate_inverted_pendulum.py \
  "${run_prefix}-h128/latest.pt" --seed 20000 --episodes 100 \
  --out "${run_prefix}-evaluation"

#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Default 4-method L63 sweep:
# 1) fixed OT, 2) learnable sinkhorn, 3) WGAN OT, 4) baseline
METHODS=${METHODS:-"fixed_ot sinkhorn wgan baseline"}
RUN_EVAL=${RUN_EVAL:-1}

for method in ${METHODS}; do
  echo "\n=== [L63] Training ${method} ==="
  METHOD="${method}" bash experiments/run_train_l63_once.sh

  if [ "${RUN_EVAL}" = "1" ]; then
    echo "=== [L63] Evaluating ${method} ==="
    METHOD="${method}" bash experiments/run_eval_l63_once.sh
  fi
done

if [ "${RUN_EVAL}" = "1" ]; then
  echo "\nFinished L63 training + evaluation for methods: ${METHODS}"
else
  echo "\nFinished L63 training for methods: ${METHODS}"
fi

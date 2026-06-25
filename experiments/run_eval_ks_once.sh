#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

METHOD=${METHOD:-}
PREFIX=${PREFIX:-}

TRAIN_SIZE=${TRAIN_SIZE:-10}
BATCH_SIZE=${BATCH_SIZE:-20}
X_LEN=${X_LEN:-100}
MODES=${MODES:-128}
WIDTH=${WIDTH:-256}
EPOCHS=${EPOCHS:-701}
X_LEN_TRAIN=${X_LEN_TRAIN:-100}
EVAL_LENGTHS=${EVAL_LENGTHS:-1000}
TRAIN_NOISY_SCALE=${TRAIN_NOISY_SCALE:-${NOISY_SCALE:-0.3}}
EVAL_NOISY_SCALE=${EVAL_NOISY_SCALE:-${NOISY_SCALE:-0.3}}
KS_NOISY_EVAL_SPLIT=${KS_NOISY_EVAL_SPLIT:-test}
KS_CLEAN_EVAL_INIT_NOISE=${KS_CLEAN_EVAL_INIT_NOISE:-0}
KS_DATA_ROOT=${KS_DATA_ROOT:-ks_data_x_single_traj/ks_single_traj_expanded_10_10}
KS_DATA_VAL=${KS_DATA_VAL:-${KS_DATA_ROOT}/ks_single_traj_val}
KS_DATA_TEST=${KS_DATA_TEST:-${KS_DATA_ROOT}/ks_single_traj_test}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-ks_output_foulder}
SUMMARY_DIM=${SUMMARY_DIM:-3}
WGAN_CRITIC_STEPS=${WGAN_CRITIC_STEPS:-2}
WGAN_CLIP=${WGAN_CLIP:-0.01}
SEED=${SEED:-42}
SKIP_CLEAN_EVAL=${SKIP_CLEAN_EVAL:-0}
SKIP_NOISY_EVAL=${SKIP_NOISY_EVAL:-0}

if [ -z "${PREFIX}" ]; then
  METHOD=${METHOD:?Either PREFIX or METHOD must be set}
  case "${METHOD}" in
    baseline)
      PREFIX="baseline_ks_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}"
      ;;
    fixed_ot)
      PREFIX="ot_fixed_ks_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}"
      ;;
    sinkhorn)
      PREFIX="sinkhorn_ks_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}"
      ;;
    wgan)
      PREFIX="wgan_ks_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_steps${WGAN_CRITIC_STEPS}_clip${WGAN_CLIP}_s${SEED}"
      ;;
    *)
      echo "Unsupported METHOD=${METHOD}" >&2
      exit 1
      ;;
  esac
fi

read -r -a EVAL_LENGTHS_ARR <<< "${EVAL_LENGTHS}"

echo "=== Evaluating prefix=${PREFIX} | train_noise=${TRAIN_NOISY_SCALE} | eval_noise=${EVAL_NOISY_SCALE} | rollouts=${EVAL_LENGTHS} ==="

eval_args=(
  --prefix "${PREFIX}" \
  --modes "${MODES}" \
  --width "${WIDTH}" \
  --gpu 0 \
  --epochs "${EPOCHS}" \
  --x_len_train "${X_LEN_TRAIN}" \
  --eval_lengths "${EVAL_LENGTHS_ARR[@]}" \
  --ks_data_val "${KS_DATA_VAL}" \
  --ks_data_test "${KS_DATA_TEST}" \
  --ks_noisy_eval_split "${KS_NOISY_EVAL_SPLIT}" \
  --ks_clean_eval_init_noise "${KS_CLEAN_EVAL_INIT_NOISE}" \
  --noisy_scale "${EVAL_NOISY_SCALE}" \
  --output_folder "${OUTPUT_FOLDER}"
)

if [ "${SKIP_CLEAN_EVAL}" = "1" ]; then
  eval_args+=(--skip_clean_eval)
fi
if [ "${SKIP_NOISY_EVAL}" = "1" ]; then
  eval_args+=(--skip_noisy_eval)
fi

exec ./.venv/bin/python -u scripts/eval_ks_standalone.py "${eval_args[@]}"

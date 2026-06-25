#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

METHOD=${METHOD:-}
PREFIX=${PREFIX:-}

TRAIN_SIZE=${TRAIN_SIZE:-100}
BATCH_SIZE=${BATCH_SIZE:-20}
X_LEN=${X_LEN:-300}
MODES=${MODES:-2}
WIDTH=${WIDTH:-256}
EPOCHS=${EPOCHS:-201}
X_LEN_TRAIN=${X_LEN_TRAIN:-${X_LEN}}
EVAL_LENGTHS=${EVAL_LENGTHS:-"100 200 500 1000"}
TRAIN_NOISY_SCALE=${TRAIN_NOISY_SCALE:-${NOISY_SCALE:-0.3}}
EVAL_NOISY_SCALE=${EVAL_NOISY_SCALE:-${NOISY_SCALE:-0.3}}
L63_NOISY_EVAL_SPLIT=${L63_NOISY_EVAL_SPLIT:-test}
L63_CLEAN_EVAL_INIT_NOISE=${L63_CLEAN_EVAL_INIT_NOISE:-0}
L63_DATA_ROOT=${L63_DATA_ROOT:-l63_data_x}
L63_DATA_VAL=${L63_DATA_VAL:-${L63_DATA_ROOT}/l63_data_val}
L63_DATA_TEST=${L63_DATA_TEST:-${L63_DATA_ROOT}/l63_data_test}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-l63_output_folder}
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
      PREFIX="baseline_l63_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}"
      ;;
    fixed_ot)
      PREFIX="ot_fixed_l63_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}"
      ;;
    sinkhorn)
      PREFIX="sinkhorn_l63_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}"
      ;;
    wgan)
      PREFIX="wgan_l63_ns${TRAIN_NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_steps${WGAN_CRITIC_STEPS}_clip${WGAN_CLIP}_s${SEED}"
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
  --l63_data_val "${L63_DATA_VAL}" \
  --l63_data_test "${L63_DATA_TEST}" \
  --l63_noisy_eval_split "${L63_NOISY_EVAL_SPLIT}" \
  --l63_clean_eval_init_noise "${L63_CLEAN_EVAL_INIT_NOISE}" \
  --noisy_scale "${EVAL_NOISY_SCALE}" \
  --output_folder "${OUTPUT_FOLDER}"
)

if [ "${SKIP_CLEAN_EVAL}" = "1" ]; then
  eval_args+=(--skip_clean_eval)
fi
if [ "${SKIP_NOISY_EVAL}" = "1" ]; then
  eval_args+=(--skip_noisy_eval)
fi

exec ./.venv/bin/python -u scripts/eval_l63_standalone.py "${eval_args[@]}"

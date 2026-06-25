#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

METHOD=${METHOD:?METHOD must be set to baseline, fixed_ot, sinkhorn, or wgan}

L63_DATA_ROOT=${L63_DATA_ROOT:-l63_data_x}
TRAIN_SIZE=${TRAIN_SIZE:-100}
BATCH_SIZE=${BATCH_SIZE:-20}
X_LEN=${X_LEN:-300}
L63_EVAL_LENGTHS=${L63_EVAL_LENGTHS:-"1000"}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-1}
L63_NOISY_EVAL_SPLIT=${L63_NOISY_EVAL_SPLIT:-test}
L63_CLEAN_EVAL_INIT_NOISE=${L63_CLEAN_EVAL_INIT_NOISE:-0}
MODES=${MODES:-2}
WIDTH=${WIDTH:-256}
EPOCHS=${EPOCHS:-701}
LEARNING_RATE=${LEARNING_RATE:-0.001}
NOISY_SCALE=${NOISY_SCALE:-0.1}
SEED=${SEED:-42}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-l63_output_folder_new}
ENABLE_WANDB=${ENABLE_WANDB:-1}

LAMBDA_OT=${LAMBDA_OT:-5}
BLUR=${BLUR:-0.02}
OT_START_EPOCH=${OT_START_EPOCH:-10}
OT_WARMUP_EPOCHS=${OT_WARMUP_EPOCHS:-20}
L63_OT_DIMS=${L63_OT_DIMS:-"0 1 2"}
L63_OT_FEATURE_MODE=${L63_OT_FEATURE_MODE:-lagged_state}
L63_OT_LAGS=${L63_OT_LAGS:-"0 1 2 4"}
OT_STD_FLOOR=${OT_STD_FLOOR:-0.01}
OT_STAT_CLIP=${OT_STAT_CLIP:-0}
OT_WEIGHTED_CAP=${OT_WEIGHTED_CAP:-0}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0}

SUMMARY_DIM=${SUMMARY_DIM:-1}
WGAN_CRITIC_STEPS=${WGAN_CRITIC_STEPS:-1}
WGAN_CLIP=${WGAN_CLIP:-0.01}
SINKHORN_SUMMARY_CLIP=${SINKHORN_SUMMARY_CLIP:-0.01}
STABLE_FIXED_OT_PRESET=${STABLE_FIXED_OT_PRESET:-1}

# Effective values; can be overridden globally or by fixed-OT preset below.
BATCHES_PER_EPOCH_EFFECTIVE=${BATCHES_PER_EPOCH}
OT_START_EPOCH_EFFECTIVE=${OT_START_EPOCH}
OT_WARMUP_EPOCHS_EFFECTIVE=${OT_WARMUP_EPOCHS}
LAMBDA_OT_EFFECTIVE=${LAMBDA_OT}
LEARNING_RATE_EFFECTIVE=${LEARNING_RATE}
OT_STD_FLOOR_EFFECTIVE=${OT_STD_FLOOR}
OT_STAT_CLIP_EFFECTIVE=${OT_STAT_CLIP}
OT_WEIGHTED_CAP_EFFECTIVE=${OT_WEIGHTED_CAP}
GRAD_CLIP_NORM_EFFECTIVE=${GRAD_CLIP_NORM}

if [ "${METHOD}" = "fixed_ot" ] && [ "${STABLE_FIXED_OT_PRESET}" = "1" ]; then
  BATCHES_PER_EPOCH_EFFECTIVE=${FIXED_OT_BATCHES_PER_EPOCH:-5}
  OT_START_EPOCH_EFFECTIVE=${FIXED_OT_OT_START_EPOCH:-100}
  OT_WARMUP_EPOCHS_EFFECTIVE=${FIXED_OT_OT_WARMUP_EPOCHS:-220}
  LAMBDA_OT_EFFECTIVE=${FIXED_OT_LAMBDA_OT:-1.5}
  LEARNING_RATE_EFFECTIVE=${FIXED_OT_LEARNING_RATE:-3e-4}
  OT_STD_FLOOR_EFFECTIVE=${FIXED_OT_OT_STD_FLOOR:-0.10}
  OT_STAT_CLIP_EFFECTIVE=${FIXED_OT_OT_STAT_CLIP:-8.0}
  OT_WEIGHTED_CAP_EFFECTIVE=${FIXED_OT_OT_WEIGHTED_CAP:-50.0}
  GRAD_CLIP_NORM_EFFECTIVE=${FIXED_OT_GRAD_CLIP_NORM:-0.0}
  echo "[stable-fixed-ot] enabled with safer defaults (override via FIXED_OT_* env vars)."
fi

read -r -a L63_EVAL_LENGTHS_ARR <<< "${L63_EVAL_LENGTHS}"
read -r -a L63_OT_DIMS_ARR <<< "${L63_OT_DIMS}"
read -r -a L63_OT_LAGS_ARR <<< "${L63_OT_LAGS}"

common_args=(
  --l63
  --l63_data_train "${L63_DATA_ROOT}/l63_data_train"
  --l63_data_val "${L63_DATA_ROOT}/l63_data_val"
  --l63_data_test "${L63_DATA_ROOT}/l63_data_test"
  --training_size "${TRAIN_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --batch_size_metricL "${BATCH_SIZE}"
  --batches_per_epoch "${BATCHES_PER_EPOCH_EFFECTIVE}"
  --modes "${MODES}"
  --width "${WIDTH}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE_EFFECTIVE}"
  --x_len "${X_LEN}"
  --l63_eval_lengths "${L63_EVAL_LENGTHS_ARR[@]}"
  --l63_noisy_eval_split "${L63_NOISY_EVAL_SPLIT}"
  --l63_clean_eval_init_noise "${L63_CLEAN_EVAL_INIT_NOISE}"
  --noisy_scale "${NOISY_SCALE}"
  --seed "${SEED}"
  --output_folder "${OUTPUT_FOLDER}"
  --train_operator
  --ot_start_epoch "${OT_START_EPOCH_EFFECTIVE}"
  --ot_warmup_epochs "${OT_WARMUP_EPOCHS_EFFECTIVE}"
  --ot_std_floor "${OT_STD_FLOOR_EFFECTIVE}"
  --ot_stat_clip "${OT_STAT_CLIP_EFFECTIVE}"
  --ot_weighted_cap "${OT_WEIGHTED_CAP_EFFECTIVE}"
  --grad_clip_norm "${GRAD_CLIP_NORM_EFFECTIVE}"
  --l63_ot_dims "${L63_OT_DIMS_ARR[@]}"
  --l63_ot_feature_mode "${L63_OT_FEATURE_MODE}"
  --l63_ot_lags "${L63_OT_LAGS_ARR[@]}"
)

if [ "${ENABLE_WANDB}" = "1" ]; then
  common_args+=(--wandb)
fi

case "${METHOD}" in
  baseline)
    PREFIX=${PREFIX:-baseline_l63_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}}
    method_args=(
      --prefix "${PREFIX}"
      --wandb_run_name "${PREFIX}"
      --with_geomloss 0
    )
    ;;
  fixed_ot)
    PREFIX=${PREFIX:-ot_fixed_l63_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}}
    method_args=(
      --prefix "${PREFIX}"
      --wandb_run_name "${PREFIX}"
      --with_geomloss 1
      --blur "${BLUR}"
      --lambda_geomloss "${LAMBDA_OT_EFFECTIVE}"
    )
    ;;
  sinkhorn)
    PREFIX=${PREFIX:-sinkhorn_l63_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}}
    method_args=(
      --prefix "${PREFIX}"
      --wandb_run_name "${PREFIX}"
      --with_geomloss 1
      --blur "${BLUR}"
      --lambda_geomloss "${LAMBDA_OT}"
      --loss_mode learnable_sinkhorn
      --wgan_critic_steps 2
      --summary_clip "${SINKHORN_SUMMARY_CLIP}"
      --summary_dim "${SUMMARY_DIM}"
      --summary_mode linear
      --state_dim 3
    )
    ;;
  wgan)
    PREFIX=${PREFIX:-wgan_l63_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_steps${WGAN_CRITIC_STEPS}_clip${WGAN_CLIP}_s${SEED}}
    method_args=(
      --prefix "${PREFIX}"
      --wandb_run_name "${PREFIX}"
      --with_geomloss 1
      --blur "${BLUR}"
      --lambda_geomloss "${LAMBDA_OT}"
      --loss_mode learnable_ot
      --wgan_critic_steps "${WGAN_CRITIC_STEPS}"
      --wgan_clip "${WGAN_CLIP}"
      --summary_dim "${SUMMARY_DIM}"
      --summary_mode linear
      --state_dim 3
    )
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 1
    ;;
esac

echo "=== Training ${METHOD} | seed=${SEED} | bs=${BATCH_SIZE} | prefix=${PREFIX} ==="

exec ./.venv/bin/python -u scripts/main.py "${common_args[@]}" "${method_args[@]}"

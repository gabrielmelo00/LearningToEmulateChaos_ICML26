#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

METHOD=${METHOD:?METHOD must be set to baseline, fixed_ot, sinkhorn, or wgan}

KS_DATA_ROOT=${KS_DATA_ROOT:-ks_data_x_single_traj/ks_single_traj_expanded_10_10}
TRAIN_SIZE=${TRAIN_SIZE:-10}
BATCH_SIZE=${BATCH_SIZE:-20}
X_LEN=${X_LEN:-100}
KS_EVAL_LENGTHS=${KS_EVAL_LENGTHS:-1000}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-1}
KS_NOISY_EVAL_SPLIT=${KS_NOISY_EVAL_SPLIT:-test}
KS_CLEAN_EVAL_INIT_NOISE=${KS_CLEAN_EVAL_INIT_NOISE:-0}
MODES=${MODES:-128}
WIDTH=${WIDTH:-256}
NOISY_SCALE=${NOISY_SCALE:-0.3}
SEED=${SEED:-42}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-ks_output_foulder}
ENABLE_WANDB=${ENABLE_WANDB:-1}

LAMBDA_OT=${LAMBDA_OT:-3}
BLUR=${BLUR:-0.02}
SUMMARY_DIM=${SUMMARY_DIM:-3}
WGAN_CRITIC_STEPS=${WGAN_CRITIC_STEPS:-2}
WGAN_CLIP=${WGAN_CLIP:-0.01}
SINKHORN_SUMMARY_CLIP=${SINKHORN_SUMMARY_CLIP:-0.01}

read -r -a KS_EVAL_LENGTHS_ARR <<< "${KS_EVAL_LENGTHS}"

common_args=(
  --kse
  --ks_data_train "${KS_DATA_ROOT}/ks_single_traj_train"
  --ks_data_val "${KS_DATA_ROOT}/ks_single_traj_val"
  --ks_data_test "${KS_DATA_ROOT}/ks_single_traj_test"
  --training_size "${TRAIN_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --batch_size_metricL "${BATCH_SIZE}"
  --batches_per_epoch "${BATCHES_PER_EPOCH}"
  --modes "${MODES}"
  --width "${WIDTH}"
  --x_len "${X_LEN}"
  --ks_eval_lengths "${KS_EVAL_LENGTHS_ARR[@]}"
  --ks_noisy_eval_split "${KS_NOISY_EVAL_SPLIT}"
  --ks_clean_eval_init_noise "${KS_CLEAN_EVAL_INIT_NOISE}"
  --noisy_scale "${NOISY_SCALE}"
  --seed "${SEED}"
  --output_folder "${OUTPUT_FOLDER}"
  --train_operator
)

if [ "${ENABLE_WANDB}" = "1" ]; then
  common_args+=(--wandb)
fi

case "${METHOD}" in
  baseline)
    PREFIX=${PREFIX:-baseline_ks_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}}
    method_args=(
      --prefix "${PREFIX}"
      --wandb_run_name "${PREFIX}"
      --with_geomloss 0
    )
    ;;
  fixed_ot)
    PREFIX=${PREFIX:-ot_fixed_ks_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}}
    method_args=(
      --prefix "${PREFIX}"
      --wandb_run_name "${PREFIX}"
      --with_geomloss 1
      --blur "${BLUR}"
      --lambda_geomloss "${LAMBDA_OT}"
    )
    ;;
  sinkhorn)
    PREFIX=${PREFIX:-sinkhorn_ks_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_s${SEED}}
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
      --summary_mode statewise
      --state_dim 256
    )
    ;;
  wgan)
    PREFIX=${PREFIX:-wgan_ks_ns${NOISY_SCALE}_xl${X_LEN}_bs${BATCH_SIZE}_ts${TRAIN_SIZE}_steps${WGAN_CRITIC_STEPS}_clip${WGAN_CLIP}_s${SEED}}
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
      --summary_mode statewise
      --state_dim 256
    )
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 1
    ;;
esac

echo "=== Training ${METHOD} | seed=${SEED} | bs=${BATCH_SIZE} | prefix=${PREFIX} ==="

exec ./.venv/bin/python -u scripts/main.py "${common_args[@]}" "${method_args[@]}"

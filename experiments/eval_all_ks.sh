#!/bin/bash
# Queue-safe KS evaluation submitter.
# Submits at most MAX_QUEUE_JOBS sweep jobs per invocation so it respects a
# 3-job profile limit. Re-run with the printed START_INDEX to submit the next wave.

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_FOLDER=${OUTPUT_FOLDER:-ks_output_foulder}
METHODS=${METHODS:-"baseline fixed_ot sinkhorn wgan"}
TRAIN_NOISE_LEVELS=${TRAIN_NOISE_LEVELS:-${NOISE_LEVELS:-"0.3"}}
SEEDS=${SEEDS:-"42 123 777"}
ROLLOUTS=${ROLLOUTS:-"100 1000"}
EVAL_NOISE_LEVELS=${EVAL_NOISE_LEVELS:-${EVAL_NOISY_SCALE:-0.3}}
KS_NOISY_EVAL_SPLIT=${KS_NOISY_EVAL_SPLIT:-test}
KS_CLEAN_EVAL_INIT_NOISE=${KS_CLEAN_EVAL_INIT_NOISE:-0}

TRAIN_SIZE=${TRAIN_SIZE:-10}
X_LEN=${X_LEN:-100}
MODES=${MODES:-128}
WIDTH=${WIDTH:-256}
EPOCHS=${EPOCHS:-701}
X_LEN_TRAIN=${X_LEN_TRAIN:-100}
BASELINE_BATCH_SIZES=${BASELINE_BATCH_SIZES:-"10 20 30 40 50"}
FIXED_OT_BATCH_SIZES=${FIXED_OT_BATCH_SIZES:-"10 20 30 40 50"}
SINKHORN_BATCH_SIZES=${SINKHORN_BATCH_SIZES:-"10 20 30 40 50"}
WGAN_BATCH_SIZES=${WGAN_BATCH_SIZES:-"10 20 30 40 50"}
SINKHORN_SUMMARY_DIM=${SINKHORN_SUMMARY_DIM:-3}
WGAN_SUMMARY_DIM=${WGAN_SUMMARY_DIM:-3}
WGAN_CRITIC_STEPS=${WGAN_CRITIC_STEPS:-2}
WGAN_CLIP=${WGAN_CLIP:-0.01}

MAX_QUEUE_JOBS=${MAX_QUEUE_JOBS:-3}
START_INDEX=${START_INDEX:-0}

batch_sizes_for_method() {
  case "$1" in
    baseline) echo "${BASELINE_BATCH_SIZES}" ;;
    fixed_ot) echo "${FIXED_OT_BATCH_SIZES}" ;;
    sinkhorn) echo "${SINKHORN_BATCH_SIZES}" ;;
    wgan) echo "${WGAN_BATCH_SIZES}" ;;
    *)
      echo "Unknown method: $1" >&2
      exit 1
      ;;
  esac
}

methods_arr=()
train_noise_arr=()
seed_arr=()

for method in ${METHODS}; do
  for train_noise in ${TRAIN_NOISE_LEVELS}; do
    for seed in ${SEEDS}; do
      methods_arr+=("${method}")
      train_noise_arr+=("${train_noise}")
      seed_arr+=("${seed}")
    done
  done
done

total=${#methods_arr[@]}
if [ "${START_INDEX}" -ge "${total}" ]; then
  echo "START_INDEX=${START_INDEX} is past the end of the queue-safe eval plan (${total} jobs)."
  exit 0
fi

submitted=0
echo "=== Submitting up to ${MAX_QUEUE_JOBS} KS eval sweep jobs starting at index ${START_INDEX} ==="

for ((i=START_INDEX; i<total && submitted<MAX_QUEUE_JOBS; i++)); do
  method=${methods_arr[i]}
  train_noise=${train_noise_arr[i]}
  seed=${seed_arr[i]}
  batch_sizes=$(batch_sizes_for_method "${method}")

  jid=$(
    METHOD="${method}" \
    SEED="${seed}" \
    TRAIN_NOISY_SCALE="${train_noise}" \
    EVAL_NOISE_LEVELS="${EVAL_NOISE_LEVELS}" \
    BATCH_SIZES="${batch_sizes}" \
    EVAL_LENGTHS="${ROLLOUTS}" \
    KS_NOISY_EVAL_SPLIT="${KS_NOISY_EVAL_SPLIT}" \
    KS_CLEAN_EVAL_INIT_NOISE="${KS_CLEAN_EVAL_INIT_NOISE}" \
    TRAIN_SIZE="${TRAIN_SIZE}" \
    X_LEN="${X_LEN}" \
    MODES="${MODES}" \
    WIDTH="${WIDTH}" \
    EPOCHS="${EPOCHS}" \
    X_LEN_TRAIN="${X_LEN_TRAIN}" \
    OUTPUT_FOLDER="${OUTPUT_FOLDER}" \
    SUMMARY_DIM=$([ "${method}" = "sinkhorn" ] && echo "${SINKHORN_SUMMARY_DIM}" || echo "${WGAN_SUMMARY_DIM}") \
    WGAN_CRITIC_STEPS="${WGAN_CRITIC_STEPS}" \
    WGAN_CLIP="${WGAN_CLIP}" \
    sbatch --parsable experiments/eval_ks_sweep_seed.sh
  )

  echo "  [${i}] method=${method} train_noise=${train_noise} seed=${seed} batch_sizes=\"${batch_sizes}\" -> job ${jid}"
  submitted=$((submitted + 1))
done

next_index=$((START_INDEX + submitted))
echo
echo "Submitted ${submitted} eval sweep jobs out of ${total} planned."
if [ "${next_index}" -lt "${total}" ]; then
  echo "Submit the next wave with:"
  echo "START_INDEX=${next_index} MAX_QUEUE_JOBS=${MAX_QUEUE_JOBS} bash experiments/eval_all_ks.sh"
else
  echo "All eval sweep jobs have been submitted."
  echo "When they finish, summarize with:"
  echo "sbatch experiments/summary_ks_job.sh"
  echo "or"
  echo "./.venv/bin/python scripts/summary_ks.py --output_folder ${OUTPUT_FOLDER}"
fi

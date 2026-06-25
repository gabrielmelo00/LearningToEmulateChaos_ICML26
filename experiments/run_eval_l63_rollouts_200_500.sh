#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_FOLDER=${OUTPUT_FOLDER:-l63_output_folder}
EVAL_LENGTHS=${EVAL_LENGTHS:-"200 500"}
# Optional explicit prefixes, e.g.:
# PREFIXES="baseline_l63_... fixed_ot_l63_..."
PREFIXES=${PREFIXES:-}

# Forwarded eval settings (same defaults as run_eval_l63_once.sh)
MODES=${MODES:-2}
WIDTH=${WIDTH:-64}
EPOCHS=${EPOCHS:-701}
X_LEN_TRAIN=${X_LEN_TRAIN:-300}
EVAL_NOISY_SCALE=${EVAL_NOISY_SCALE:-0.3}
L63_NOISY_EVAL_SPLIT=${L63_NOISY_EVAL_SPLIT:-test}
L63_CLEAN_EVAL_INIT_NOISE=${L63_CLEAN_EVAL_INIT_NOISE:-0}
L63_DATA_ROOT=${L63_DATA_ROOT:-l63_data_x}
L63_DATA_VAL=${L63_DATA_VAL:-${L63_DATA_ROOT}/l63_data_val}
L63_DATA_TEST=${L63_DATA_TEST:-${L63_DATA_ROOT}/l63_data_test}
SKIP_CLEAN_EVAL=${SKIP_CLEAN_EVAL:-0}
SKIP_NOISY_EVAL=${SKIP_NOISY_EVAL:-0}

if [ -n "${PREFIXES}" ]; then
  read -r -a PREFIX_ARR <<< "${PREFIXES}"
else
  PREFIX_ARR=()
  while IFS= read -r dir; do
    prefix=$(basename "${dir}")
    eval_root="${dir}/eval_noisy_trainval_clean_test"
    if [ -d "${eval_root}/rollout_100" ] && [ -d "${eval_root}/rollout_1000" ]; then
      PREFIX_ARR+=("${prefix}")
    fi
  done < <(find "${OUTPUT_FOLDER}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [ ${#PREFIX_ARR[@]} -eq 0 ]; then
  echo "No prefixes found to evaluate in ${OUTPUT_FOLDER}."
  echo "Tip: set PREFIXES=\"<prefix1> <prefix2>\" explicitly."
  exit 1
fi

echo "=== Running L63 eval for rollouts: ${EVAL_LENGTHS} ==="
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Found ${#PREFIX_ARR[@]} prefix(es)."

for prefix in "${PREFIX_ARR[@]}"; do
  echo
  echo "--- Evaluating prefix=${prefix} (rollouts ${EVAL_LENGTHS}) ---"
  PREFIX="${prefix}" \
  EVAL_LENGTHS="${EVAL_LENGTHS}" \
  MODES="${MODES}" \
  WIDTH="${WIDTH}" \
  EPOCHS="${EPOCHS}" \
  X_LEN_TRAIN="${X_LEN_TRAIN}" \
  EVAL_NOISY_SCALE="${EVAL_NOISY_SCALE}" \
  L63_NOISY_EVAL_SPLIT="${L63_NOISY_EVAL_SPLIT}" \
  L63_CLEAN_EVAL_INIT_NOISE="${L63_CLEAN_EVAL_INIT_NOISE}" \
  L63_DATA_VAL="${L63_DATA_VAL}" \
  L63_DATA_TEST="${L63_DATA_TEST}" \
  OUTPUT_FOLDER="${OUTPUT_FOLDER}" \
  SKIP_CLEAN_EVAL="${SKIP_CLEAN_EVAL}" \
  SKIP_NOISY_EVAL="${SKIP_NOISY_EVAL}" \
  bash experiments/run_eval_l63_once.sh

done

echo
echo "Done. Added rollout_200 and rollout_500 results under each prefix in ${OUTPUT_FOLDER}."

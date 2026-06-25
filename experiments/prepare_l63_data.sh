#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-./.venv/bin/python}
L63_DATA_ROOT=${L63_DATA_ROOT:-l63_data_x}
NOISY_SCALE=${NOISY_SCALE:-0.3}
OVERWRITE_NOISE=${OVERWRITE_NOISE:-0}

N_TRAIN=${N_TRAIN:-100}
N_VAL=${N_VAL:-20}
N_TEST=${N_TEST:-20}
DT=${DT:-0.01}
N_STEPS=${N_STEPS:-10000}
STRIDE=${STRIDE:-5}
BURNIN_STEPS=${BURNIN_STEPS:-5000}
N_WORKERS=${N_WORKERS:-8}

${PYTHON_BIN} l63_data_x/generate_data.py \
  --output_root "${L63_DATA_ROOT}" \
  --n_train "${N_TRAIN}" \
  --n_val "${N_VAL}" \
  --n_test "${N_TEST}" \
  --dt "${DT}" \
  --n_steps "${N_STEPS}" \
  --stride "${STRIDE}" \
  --burnin_steps "${BURNIN_STEPS}" \
  --n_workers "${N_WORKERS}"

noise_args=(
  --data_path "${L63_DATA_ROOT}/l63_data_train"
  --data_path "${L63_DATA_ROOT}/l63_data_val"
  --data_path "${L63_DATA_ROOT}/l63_data_test"
  --noisy_scale "${NOISY_SCALE}"
)

if [ "${OVERWRITE_NOISE}" = "1" ]; then
  noise_args+=(--overwrite)
fi

${PYTHON_BIN} dataloader/dataloader_l63.py "${noise_args[@]}"

echo "Prepared L63 data at ${L63_DATA_ROOT} (noise scale ${NOISY_SCALE})."

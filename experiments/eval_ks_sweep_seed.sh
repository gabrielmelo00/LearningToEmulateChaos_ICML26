#!/bin/bash
# One GPU job that evaluates a single method for one seed across batch sizes and eval noise levels.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ks_eval_sweep_seed
#SBATCH -p gpu-x
#SBATCH --time=168:00:00
#SBATCH --gpus=1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
module purge
module load hwloc
module load cuda

export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${SLURM_JOB_ID}}

METHOD=${METHOD:?METHOD must be set to baseline, fixed_ot, sinkhorn, or wgan}
SEED=${SEED:?SEED must be set}
BATCH_SIZES=${BATCH_SIZES:-"10 20 30 40 50"}
TRAIN_NOISY_SCALE=${TRAIN_NOISY_SCALE:-0.3}
EVAL_NOISE_LEVELS=${EVAL_NOISE_LEVELS:-0.3}

for bs in ${BATCH_SIZES}; do
  echo
  echo "=== Sweep eval | method=${METHOD} | seed=${SEED} | batch_size=${bs} | clean branch ==="
  METHOD="${METHOD}" SEED="${SEED}" BATCH_SIZE="${bs}" \
    TRAIN_NOISY_SCALE="${TRAIN_NOISY_SCALE}" SKIP_NOISY_EVAL=1 \
    bash experiments/run_eval_ks_once.sh

  for eval_noise in ${EVAL_NOISE_LEVELS}; do
    echo
    echo "=== Sweep eval | method=${METHOD} | seed=${SEED} | batch_size=${bs} | eval_noise=${eval_noise} ==="
    METHOD="${METHOD}" SEED="${SEED}" BATCH_SIZE="${bs}" \
      TRAIN_NOISY_SCALE="${TRAIN_NOISY_SCALE}" EVAL_NOISY_SCALE="${eval_noise}" SKIP_CLEAN_EVAL=1 \
      bash experiments/run_eval_ks_once.sh
  done
done

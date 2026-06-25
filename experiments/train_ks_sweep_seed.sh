#!/bin/bash
# One GPU job that trains a single method for one seed across multiple batch sizes.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ks_train_sweep_seed
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
export NCCL_DEBUG=info
export CUDA_LAUNCH_BLOCKING=1
export PYKEOPS_VERBOSE=${PYKEOPS_VERBOSE:-1}

METHOD=${METHOD:?METHOD must be set to baseline, fixed_ot, sinkhorn, or wgan}
SEED=${SEED:?SEED must be set}
BATCH_SIZES=${BATCH_SIZES:-"10 20 30 40 50"}

for bs in ${BATCH_SIZES}; do
  echo
  echo "=== Sweep train | method=${METHOD} | seed=${SEED} | batch_size=${bs} ==="
  METHOD="${METHOD}" SEED="${SEED}" BATCH_SIZE="${bs}" \
    bash experiments/run_train_ks_once.sh
done

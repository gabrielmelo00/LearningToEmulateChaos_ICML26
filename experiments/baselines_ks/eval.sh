#!/bin/bash
# Evaluate an already-trained KS baseline checkpoint with configurable rollout lengths.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseline_ks_eval
#SBATCH -p gpu-x
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
module purge
module load hwloc
module load cuda

export OMP_NUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${SLURM_JOB_ID}}

bash experiments/run_eval_ks_once.sh

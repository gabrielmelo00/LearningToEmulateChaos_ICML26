#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseline_ks_1gpu
#SBATCH -p gpu-x
#SBATCH --time=22:00:00
#SBATCH --gpus=1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
module purge
module load hwloc
module load cuda

export OMP_NUM_THREADS=1
export NCCL_DEBUG=info
export CUDA_LAUNCH_BLOCKING=1

METHOD=baseline bash experiments/run_train_ks_once.sh

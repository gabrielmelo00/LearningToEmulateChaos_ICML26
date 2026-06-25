#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ot_ks_eval_partial
#SBATCH -p gpu-x
#SBATCH --time=04:00:00
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
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun --ntasks=1 --gres=gpu:1 --partition=gpu-x ./.venv/bin/python -u scripts/main.py \
  --kse \
  --batch_size 25 \
  --modes 28 \
  --width 256 \
  --x_len 100 \
  --with_geomloss 1 \
  --blur 0.02 \
  --lambda_geomloss 3

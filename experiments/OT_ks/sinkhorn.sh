#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ot_ks_sinkhorn
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
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYKEOPS_VERBOSE=1

rm -rf ~/.cache/keops*

srun --ntasks=1 --gres=gpu:1 --partition=gpu-x ./.venv/bin/python -u scripts/main.py \
  --kse \
  --batch_size 5 \
  --modes 28 \
  --width 256 \
  --x_len 100 \
  --with_geomloss 1 \
  --blur 0.02 \
  --lambda_geomloss 3 \
  --prefix "state_sinkhorn_dim4_pointwise_ks_1gpu" \
  --train_operator \
  --wandb \
  --loss_mode learnable_sinkhorn \
  --wgan_critic_steps 2 \
  --summary_clip 0.1 \
  --summary_dim 4 \
  --summary_mode pointwise \
  --state_dim 256

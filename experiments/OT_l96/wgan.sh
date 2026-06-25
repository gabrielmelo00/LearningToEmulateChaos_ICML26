#!/bin/bash
#SBATCH --job-name=test_l96
#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

cd "$(dirname "$0")/../.."

export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Clear PyKeOps cache to avoid GPU architecture incompatibility
rm -rf ~/.cache/keops*
export PYKEOPS_VERBOSE=1

EXP_NAME=${1:-dim1wgan_n_2gpus}



while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} scripts/main.py \
  --l96 \
  --batch_size 25 \
  --modes 28 \
  --width 64 \
  --x_len 100 \
  --with_geomloss_kd 0 \
  --with_geomloss 1 \
  --blur 0.02 \
  --lambda_geomloss 3 \
  --noisy_scale 0.3 \
  --prefix "state_wgan_dim3_mlp_1gpu_" \
  --train_operator \
  --wandb \
  --loss_mode learnable_ot \
  --wgan_critic_steps 5 \
  --wgan_clip 0.1 \
  --summary_dim 3 \
  --summary_mode statewise \ 
  #--state_dim 60 \
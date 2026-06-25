#!/bin/bash
#SBATCH --job-name=lorentz63-1000
#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=22:00:00
#SBATCH --partition=CPU
#SBATCH --gpus=0

python dataloader/dataloader_ks.py \
  --data_path ks_data_x_single_traj/ks_single_traj_train \
  --data_path ks_data_x_single_traj/ks_single_traj_val \
  --data_path ks_data_x_single_traj/ks_single_traj_test \
  --noisy_scale 0.3 

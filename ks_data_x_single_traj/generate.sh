#!/bin/bash
#SBATCH --job-name=ks-single-traj
#SBATCH --output=logs/gen%j.log
#SBATCH --error=logs/gen%j.err
#SBATCH --time=22:00:00
#SBATCH --partition=par48-x
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/users/l201292/DynSysOT/ks_data_x_single_traj
/home/users/l201292/DynSysOT/.venv/bin/python generate_data_single_traj.py

#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --job-name=gen_ks_data
#SBATCH --partition=serial
#SBATCH -o gen_ks_data.%j.out
#SBATCH -e gen_ks_data.%j.err

cd $SLURM_SUBMIT_DIR
module purge
module load hwloc
module load gnu12
module load openmpi4

./.venv/bin/python ks_data_x_single_traj/generate_data_single_traj.py

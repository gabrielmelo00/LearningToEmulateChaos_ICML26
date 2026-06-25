#!/bin/bash
# GPU-backed summary job for clusters that require GPU requests on gpu-x.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ks_summary
#SBATCH -p gpu-x
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
module purge
module load hwloc
module load cuda

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${SLURM_JOB_ID}}

OUTPUT_FOLDER=${OUTPUT_FOLDER:-ks_output_foulder}
SUMMARY_ARGS=${SUMMARY_ARGS:-}

./.venv/bin/python scripts/summary_ks.py --output_folder "${OUTPUT_FOLDER}" ${SUMMARY_ARGS}

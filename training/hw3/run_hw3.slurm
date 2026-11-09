#!/bin/bash
#SBATCH --job-name=ds7400-hw3
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer/logs/hw3-%j.out
set -euo pipefail
module purge
module load gcc/14.2.0 python/3.13.11
source /home/thq4up/courses/DS7400/.venv-hw3/bin/activate
cd /home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/thq4up-hw3-${SLURM_JOB_ID}}/matplotlib"
export OMP_NUM_THREADS=8
mkdir -p "$MPLCONFIGDIR"
python -u -m training.hw3.train_experiments --output-dir "artifacts_hw3/run-${SLURM_JOB_ID}"

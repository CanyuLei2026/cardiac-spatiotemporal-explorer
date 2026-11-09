#!/bin/bash
#SBATCH --job-name=hw3-test
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer/logs/hw3-test-%j.out
set -euo pipefail
module purge
module load gcc/14.2.0 python/3.13.11
source /home/thq4up/courses/DS7400/.venv-hw3/bin/activate
cd /home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/thq4up-hw3-${SLURM_JOB_ID}}/matplotlib"
export OMP_NUM_THREADS=8
mkdir -p "$MPLCONFIGDIR"
python -u -m training.hw3.evaluate_selected --output-dir "artifacts_hw3/test-${SLURM_JOB_ID}"

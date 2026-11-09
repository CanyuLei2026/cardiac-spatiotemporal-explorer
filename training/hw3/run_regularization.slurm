#!/bin/bash
#SBATCH --job-name=hw3-context
#SBATCH --partition=standard
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer/logs/hw3-context-%A-%a.out
set -euo pipefail
module purge
module load gcc/14.2.0 python/3.13.11
source /home/thq4up/courses/DS7400/.venv-hw3/bin/activate
cd /home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/thq4up-hw3-${SLURM_JOB_ID}}/matplotlib"
export OMP_NUM_THREADS=8
mkdir -p "$MPLCONFIGDIR"
HW3_SEED=$((7400 + SLURM_ARRAY_TASK_ID))
python -u -m training.hw3.train_experiments --config training/hw3/regularization_config.json --seed "$HW3_SEED" --validation-only --output-dir "artifacts_hw3/context-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

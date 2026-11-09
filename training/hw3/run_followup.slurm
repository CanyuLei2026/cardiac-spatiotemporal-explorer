#!/bin/bash
#SBATCH --job-name=hw3-followup
#SBATCH --partition=standard
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer/logs/hw3-followup-%A-%a.out
set -euo pipefail
module purge
module load gcc/14.2.0 python/3.13.11
source /home/thq4up/courses/DS7400/.venv-hw3/bin/activate
cd /home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/thq4up-hw3-${SLURM_JOB_ID}}/matplotlib"
export OMP_NUM_THREADS=8
mkdir -p "$MPLCONFIGDIR"
case "$SLURM_ARRAY_TASK_ID" in
  0) HW3_LR=0.0001 ;;
  1) HW3_LR=0.00003 ;;
  *) exit 2 ;;
esac
python -u -m training.hw3.train_experiments --epochs 60 --learning-rate "$HW3_LR" --validation-only --output-dir "artifacts_hw3/followup-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

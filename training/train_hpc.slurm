#!/bin/bash
#SBATCH --job-name=ds7400-hw2-mlp
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer/logs/train-%j.out

set -euo pipefail

module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
source /home/thq4up/courses/DS7400/.venv/bin/activate

PROJECT_DIR=/home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer
DATA_DIR=/project/jxie_group/Canyu/B2H/data/pointcloud_mat_9.16
cd "$PROJECT_DIR"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/${USER}-${SLURM_JOB_ID}}/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$MPLCONFIGDIR" artifacts public/model logs

python -u -m training.train \
  --data-dir "$DATA_DIR" \
  --train-cases case_0001 case_0002 case_0003 \
  --val-case case_0005 \
  --test-case case_0004 \
  --output-dir artifacts \
  --browser-model public/model/activation_model.json \
  --epochs 50 \
  --patience 7

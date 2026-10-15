# Activation classifier training

This directory trains the small PyTorch MLP used by the web application. The binary task is: **will the selected point's steepest positive `U_heart` activation occur within the next 20 frames?**

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/requirements.txt
```

## Five-case experiment

Run from the repository root. Replace the example case IDs only if the data audit finds an incomplete case.

```bash
python -m training.audit_cases \
  /project/jxie_group/Canyu/B2H/data/pointcloud_mat_9.16 \
  --limit 10
```

```bash
python -m training.train \
  --data-dir /project/jxie_group/Canyu/B2H/data/pointcloud_mat_9.16 \
  --train-cases case_0001 case_0002 case_0003 \
  --val-case case_0005 \
  --test-case case_0004
```

The script performs a case-level split before generating windows, computes normalization statistics from training cases only, trains both the MLP and a single-linear-layer baseline, evaluates the best saved MLP checkpoint on the training, validation, and held-out test sets, and writes:

- `artifacts/activation_mlp.pt`
- `artifacts/metrics.json`
- `artifacts/case_split.json`
- `artifacts/loss_curves.png`
- `artifacts/confusion_matrix.png`
- `public/model/activation_model.json`

The JSON file contains the exact trained weights used for browser inference.

## UVA HPC environment and Slurm job

The project uses a private CPU-only virtual environment under `/home/thq4up/courses/DS7400/.venv`. Training runs on a Slurm compute node rather than the login node:

```bash
cd /home/thq4up/courses/DS7400/cardiac-spatiotemporal-explorer
sbatch training/train_hpc.slurm
```

The job uses three training cases, `case_0005` for validation, and the unseen `case_0004` for final testing and web demonstration.

## Local development check

This command tests the pipeline using point-level splits of the compact web case. Its output is a development prototype and must not be reported as the final held-out-case result.

```bash
python -m training.train \
  --prototype-web-case public/data/case_0004 \
  --output-dir training/runs/prototype \
  --epochs 30 \
  --patience 5
```

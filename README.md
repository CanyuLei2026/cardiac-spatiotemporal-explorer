# Cardiac Spatiotemporal Explorer

UVA Fall 2026 DS7400 — Homework 1, 2, and 3

This project combines an interactive cardiac point-cloud viewer with electrical-signal classification. HW1 builds the viewer, HW2 adds an MLP and interactive training, and HW3 investigates gated fusion and context regularization.

[Open the live application](https://canyulei2026.github.io/cardiac-spatiotemporal-explorer/)

The live application currently uses the HW2 model. HW3 models are evaluated through the experiment scripts described below.

## HW1 and HW2 application

### Features

- Rotate, zoom, filter, and animate a 3D cardiac point cloud.
- Select a point to inspect its coordinates and linked electrical signal.
- Save point annotations, review notes in a list, delete markers, and import/export JSON. Annotations are stored in the browser.
- Predict activation in the next 20 frames, display confidence and ground truth, and reveal the signal after prediction.
- View model metrics, confusion matrices, and recorded training-loss replay.
- Train a small browser model with adjustable learning rate, epochs, and hidden units, with live loss/accuracy charts.

The demonstration contains a fixed-seed, 2,500-point sample from the 11,000 points in case_0004, with 801 signal frames. Point-type and surface-region labels have readable definitions in the interface. Time is displayed as simulation time because the source does not reliably identify its physical unit.

### HW2 model results

The HW2 MLP uses `44 -> 64 -> 32 -> 2`, ReLU, cross-entropy, Adam, and early stopping. Its held-out test accuracy is **94.31%**, with **5.69% error**, compared with **6.29% error** for the linear benchmark. These are the original HW2 results; the controlled HW3 reruns below use a different training budget.

The browser loads the exported weights from [activation_model.json](public/model/activation_model.json). The browser-training sandbox is a separate small experiment. Prediction and browser training require no backend or database.

### Run the web application

Requires Node.js 18 or newer:

```bash
npm install
npm run dev
```

For a production preview:

```bash
npm run build
npm run preview
```

Major libraries: React, Vite, Three.js, React Three Fiber, Drei, PyTorch, NumPy, SciPy, h5py, and Matplotlib. The full HW2 training instructions are in [training/README.md](training/README.md).

## HW3 experiment summary

### 1 Task and dataset

Predict whether a cardiac point will activate within the next 20 frames. Activation is the frame with the largest positive change in its `U_heart` signal. Inputs contain the previous 40 signal values, XYZ coordinates, and normalized current time (44 features). Features are standardized using training-set statistics.

| Split | Cases | Samples |
|---|---|---:|
| Training | case_0001, case_0002, case_0003 | 116,748 |
| Validation | case_0005 | 39,240 |
| Test | case_0004 | 42,964 |

The original dataset is too large to include. A representative 2,500-point sample of case_0004 is included for the web demo and code checks. Formal experiments require the five full MAT files containing `P_myo`, `U_heart`, and `time`.

### 2 Baseline architecture

The HW2 baseline is an MLP trained with cross-entropy loss and Adam:

```text
44 inputs -> 64 units + ReLU -> 32 units + ReLU -> 2 logits
```

### 3 Proposed modification

The hypothesis is that separate signal and context processing, followed by learned weighting, may improve prediction.

- **Dual branch:** signal history maps to 32 ReLU features `s`; XYZ/time maps to 32 ReLU features `c`. Concatenate them and apply a `64 -> 32 + ReLU -> 2` classifier.
- **Gated fusion:** compute `g = sigmoid(W_g [s,c] + b_g)` and feed `[2g * s, 2(1-g) * c]` to the classifier. Products are elementwise. The gate starts at 0.5.
- **Final model:** gated fusion with weight decay `0.001` and context dropout `0.5`. During training, the entire context embedding is removed per sample with 50% probability; retained embeddings are scaled by two. Dropout is disabled during evaluation.

Additional experiments compared longer training, a lower learning rate, different dropout probabilities, and a signal residual path.

### 4 Running the experiments

Run from the repository root with Python 3.13 and CPU dependencies:

```bash
python3.13 -m venv .venv-hw3
source .venv-hw3/bin/activate
python -m pip install -r training/hw3/requirements-hpc.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Set `dataDir` in `training/hw3/config.json` and `training/hw3/regularization_config.json` to the full MAT dataset directory. Use a new output directory for each run.

```bash
# Initial comparison
python -m training.hw3.train_experiments --output-dir artifacts_hw3/initial

# Training budget and learning-rate comparisons
python -m training.hw3.train_experiments --epochs 60 --learning-rate 0.0001 --validation-only --output-dir artifacts_hw3/lr1e4
python -m training.hw3.train_experiments --epochs 60 --learning-rate 0.00003 --validation-only --output-dir artifacts_hw3/lr3e5

# Regularization ablations
python -m training.hw3.train_experiments --config training/hw3/regularization_config.json --validation-only --output-dir artifacts_hw3/regularization

# Test the selected checkpoint
python -m training.hw3.evaluate_selected --checkpoint artifacts_hw3/regularization/gated_context50-seed7400/best_model.pt --output-dir artifacts_hw3/test7400
```

For testing elsewhere, update the case_0004 MAT path in `training/hw3/evaluate_selected.py`. Repeat evaluation for seeds 7401/7402 and the reference checkpoints. Runs save configurations, weights, metrics, and plots. UVA batch scripts are in `training/hw3/run_*.slurm`; adapt their project/environment paths before submission.

### 5 Main results

The first 20-epoch experiment averaged 93.65% test accuracy for the baseline, 90.59% for dual branches, and 91.28% for gated fusion. The original design therefore failed to improve accuracy.

The final comparison uses 60 epochs, Adam learning rate 0.0001, batch size 256, and three seeds (7400–7402). Checkpoints are selected by minimum validation loss; the classification threshold is fixed at 0.5.

| Model | Mean test accuracy | Mean F1 | Mean AUROC | Accuracy sample SD |
|---|---:|---:|---:|---:|
| Baseline | 93.83% | 0.9374 | 0.9884 | 0.57 pp |
| Original gated fusion | 89.32% | 0.9038 | 0.9883 | 2.85 pp |
| **Gated fusion with regularization** | **94.77%** | **0.9500** | **0.9918** | **0.21 pp** |

`pp` means percentage points. The final model improves average accuracy by 0.95 pp over the baseline and reduces seed variability.

### 6 What I learned

Adding branches and a gate did not automatically improve generalization. Longer training and a lower learning rate did not consistently fix instability. Weight decay and context dropout helped more, supporting context regularization on this case split, without proving a particular shortcut mechanism.

Only one validation case and one test case were used. Early test results informed later investigation, so the final test comparison is exploratory. New case splits would be needed to assess broader generalization.

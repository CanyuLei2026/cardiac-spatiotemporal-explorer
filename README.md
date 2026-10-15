# Cardiac Spatiotemporal Explorer

An interactive web application for cardiac geometry, per-point electrical signals, and neural-network activation prediction. The application links a 3D point-cloud view with an 801-frame `U_heart` trace and predicts whether the selected point will activate within the next 20 frames.

## Live demo

[Open the Cardiac Spatiotemporal Explorer](https://canyulei2026.github.io/cardiac-spatiotemporal-explorer/)

## Data

The original multi-case research dataset is stored on the UVA HPC and is too large to include in this public repository. Five complete cases are used for the final HW2 training, validation, and held-out testing experiment. `case_0004.mat` is reserved as the unseen test/demo case. To keep the web application lightweight, the included browser data is a fixed-seed, 2,500-point sample from the 11,000 cardiac points in that case.

Included variables:

- 3D myocardial point coordinates
- Per-point `U_heart` signals
- Time values
- Surface-region, point-type, and component labels
- Original point indices for traceability

## Features

- Rotate, pan, and zoom the 3D cardiac point cloud
- Color the geometry by electrical potential at the current time
- Play, pause, scrub, and change animation speed
- Select a point and inspect its signal after requesting a prediction
- Synchronize the signal cursor with the 3D time frame
- Filter points by surface region, point type, or component
- Read human-readable definitions for every numeric point label
- Switch between adaptive per-frame contrast and a fixed full-case color scale
- Add, remove, import, and export point annotations
- Browse saved annotations in a list, review notes, select their 3D points, and delete records directly
- Responsive desktop and mobile layouts
- Predict whether a selected point will activate within the next 20 frames
- Enter XYZ coordinates and simulation time; the app matches the nearest sampled point for prediction
- Run the trained PyTorch MLP weights directly in the browser
- Compare the prediction with ground truth and display confidence
- Hide the complete signal and its vertical scale while selecting a time, show only a fixed horizontal guide, then reveal the full trace and ground truth after prediction
- Highlight the 40-frame input window, 20-frame forecast horizon, and true activation
- Display training/validation loss, held-out metrics, confusion matrix, and linear-baseline comparison

## Neural-network task

For each point, the activation frame is defined as the frame with the steepest positive `U_heart` change. Each classification example contains the previous 40 signal values, normalized XYZ coordinates, and current normalized time. The binary target indicates whether activation occurs during the next 20 frames.

The MLP architecture is `44 → 64 → 32 → 2` with ReLU activations. Training uses cross-entropy loss, Adam, early stopping, and balanced positive/negative windows. A single-linear-layer logistic-regression classifier is trained on the same input as the shallow benchmark.

Cases are split before windows are generated:

- Three complete cases for training
- One complete case for validation
- One complete case for held-out testing

This prevents similar windows from the same cardiac simulation from appearing in both training and testing. See [`training/README.md`](training/README.md) for commands and generated artifacts.

## Held-out results

The final model was trained with a `0.0001` learning rate and evaluated on 42,964 examples from the unseen `case_0004`. The MLP achieved **94.31% accuracy**, **5.69% error**, **0.943 F1**, and **0.988 AUROC**. The logistic-regression benchmark achieved **93.71% accuracy** and **6.29% error** on the same examples. Thus, the neural network reduced the benchmark error by about 9.7% relative to logistic regression.

The saved metrics, learning curves, confusion matrix, and PyTorch checkpoint are in [`artifacts/`](artifacts/). The browser uses the same trained weights exported to [`public/model/activation_model.json`](public/model/activation_model.json).

## Run locally

Requires Node.js 18 or newer.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite.

To verify the production build:

```bash
npm run build
npm run preview
```

## Major libraries

- React
- Vite
- Three.js
- React Three Fiber
- Drei
- PyTorch
- NumPy, SciPy, and h5py
- Matplotlib

## Annotation and backend status

- Interactive point annotation: implemented. Annotations are stored locally in the browser and can be exported or imported as JSON.
- Neural-network inference: implemented in the browser using JSON-exported PyTorch weights.
- Backend/database: not implemented. No backend is required for prediction.

## Label definitions

- Point type: `1` interior sample, `2` endocardial surface, `3` epicardial surface, `4` full-boundary sample.
- Surface region: `0` myocardial interior, `1` outer surface, `2` LV endocardium, `3` RV endocardium.
- Component ID: connected component of the point-cloud k-NN graph. The included case contains one main component (`1`).

The source stores time from 0 to 80 with a step of 0.1, but does not reliably encode whether the unit is milliseconds or another simulation-time unit. The interface therefore uses the term “simulation time.”

## Re-export a sample case

The MATLAB converter is available at `scripts/export_case_for_web.m`. It reads the source `.mat` file without modifying it and emits compact browser assets using a fixed random seed.

## AI-assisted development disclosure

OpenAI Codex was used to assist with project planning, implementation, debugging, documentation, and interface development. The model architecture, data split, label definition, metrics, and complete training/inference workflow are documented so they can be reviewed and explained independently.

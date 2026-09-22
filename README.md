# Cardiac Spatiotemporal Explorer

An interactive web viewer for cardiac geometry and per-point electrical signals. The application links a 3D point-cloud view with an 801-frame `U_heart` trace.

## Data

The original research dataset is too large to include in this GitHub repository. Therefore, `case_0004.mat` was selected as a representative cardiac data case for this project. To keep the web demonstration lightweight, the included data is a fixed-seed, 2,500-point sample from the 11,000 cardiac points in the source case.

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
- Select a point and inspect its complete signal
- Synchronize the signal cursor with the 3D time frame
- Filter points by surface region, point type, or component
- Read human-readable definitions for every numeric point label
- Switch between adaptive per-frame contrast and a fixed full-case color scale
- Add, remove, import, and export point annotations
- Browse saved annotations in a list, review notes, select their 3D points, and delete records directly
- Responsive desktop and mobile layouts

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

## Annotation and backend status

- Interactive point annotation: implemented. Annotations are stored locally in the browser and can be exported or imported as JSON.
- Backend/database: not implemented in the initial version.

## Label definitions

- Point type: `1` interior sample, `2` endocardial surface, `3` epicardial surface, `4` full-boundary sample.
- Surface region: `0` myocardial interior, `1` outer surface, `2` LV endocardium, `3` RV endocardium.
- Component ID: connected component of the point-cloud k-NN graph. The included case contains one main component (`1`).

The source stores time from 0 to 80 with a step of 0.1, but does not reliably encode whether the unit is milliseconds or another simulation-time unit. The interface therefore uses the term “simulation time.”

## Re-export a sample case

The MATLAB converter is available at `scripts/export_case_for_web.m`. It reads the source `.mat` file without modifying it and emits compact browser assets using a fixed random seed.

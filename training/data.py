from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HISTORY_FRAMES = 40
HORIZON_FRAMES = 20
HARD_NEGATIVE_DISTANCE = 60


@dataclass(frozen=True)
class CaseArrays:
    case_id: str
    points: np.ndarray
    signals: np.ndarray
    time: np.ndarray


def _orient_case(points: np.ndarray, signals: np.ndarray, time: np.ndarray, case_id: str) -> CaseArrays:
    points = np.asarray(points, dtype=np.float32).squeeze()
    signals = np.asarray(signals, dtype=np.float32).squeeze()
    time = np.asarray(time, dtype=np.float32).reshape(-1)

    if points.ndim != 2:
        raise ValueError(f"{case_id}: P_myo must be a 2D array")
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    if points.shape[1] != 3:
        raise ValueError(f"{case_id}: P_myo has unexpected shape {points.shape}")

    if signals.ndim != 2:
        raise ValueError(f"{case_id}: U_heart must be a 2D array")
    if signals.shape[0] != points.shape[0] and signals.shape[1] == points.shape[0]:
        signals = signals.T
    if signals.shape[0] != points.shape[0]:
        raise ValueError(
            f"{case_id}: point count {points.shape[0]} does not match U_heart shape {signals.shape}"
        )
    if signals.shape[1] != time.size:
        raise ValueError(f"{case_id}: time length {time.size} does not match U_heart shape {signals.shape}")
    if time.size < HISTORY_FRAMES + HORIZON_FRAMES:
        raise ValueError(f"{case_id}: not enough time frames for the configured prediction window")

    return CaseArrays(case_id, np.ascontiguousarray(points), np.ascontiguousarray(signals), time)


def load_mat_case(path: str | Path) -> CaseArrays:
    """Load either a classic MAT file or a MATLAB v7.3 HDF5 MAT file."""
    path = Path(path)
    case_id = path.stem
    try:
        from scipy.io import loadmat

        payload = loadmat(path, variable_names=["P_myo", "U_heart", "time"])
        return _orient_case(payload["P_myo"], payload["U_heart"], payload["time"], case_id)
    except (NotImplementedError, ValueError, OSError, KeyError):
        import h5py

        with h5py.File(path, "r") as handle:
            return _orient_case(handle["P_myo"][:], handle["U_heart"][:], handle["time"][:], case_id)


def load_web_case(directory: str | Path) -> CaseArrays:
    """Load the compact binary format used by the React demonstration."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    point_count = int(metadata["pointCount"])
    time_count = int(metadata["timeCount"])
    points = np.fromfile(directory / "points.f32", dtype="<f4").reshape(point_count, 3)
    signals = np.fromfile(directory / "signals.f32", dtype="<f4").reshape(point_count, time_count)
    return _orient_case(points, signals, np.asarray(metadata["time"]), metadata["caseId"])


def subset_case(case: CaseArrays, indices: np.ndarray, suffix: str) -> CaseArrays:
    return CaseArrays(
        case_id=f"{case.case_id}-{suffix}",
        points=np.ascontiguousarray(case.points[indices]),
        signals=np.ascontiguousarray(case.signals[indices]),
        time=case.time,
    )


def activation_frames(signals: np.ndarray) -> np.ndarray:
    """Frame of the steepest positive U_heart change for every point."""
    return np.argmax(np.diff(signals, axis=1), axis=1).astype(np.int32) + 1


def _sample_without_replacement(candidates: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    if candidates.size == 0:
        return []
    size = min(count, candidates.size)
    return rng.choice(candidates, size=size, replace=False).astype(int).tolist()


def make_examples(case: CaseArrays, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create two positive and two negative windows per usable point."""
    rng = np.random.default_rng(seed)
    last_current_frame = case.signals.shape[1] - HORIZON_FRAMES - 1
    first_current_frame = HISTORY_FRAMES - 1
    activations = activation_frames(case.signals)
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for point_index, activation in enumerate(activations):
        positive = np.arange(
            max(first_current_frame, activation - HORIZON_FRAMES),
            min(last_current_frame, activation - 1) + 1,
            dtype=np.int32,
        )
        hard_negative = np.arange(
            max(first_current_frame, activation - HARD_NEGATIVE_DISTANCE),
            min(last_current_frame, activation - HORIZON_FRAMES - 1) + 1,
            dtype=np.int32,
        )
        regular_before = np.arange(
            first_current_frame,
            min(last_current_frame, activation - HARD_NEGATIVE_DISTANCE - 1) + 1,
            dtype=np.int32,
        )
        regular_after = np.arange(
            max(first_current_frame, activation),
            last_current_frame + 1,
            dtype=np.int32,
        )
        regular_negative = regular_before if regular_before.size else regular_after

        selected_positive = _sample_without_replacement(positive, 2, rng)
        selected_negative = _sample_without_replacement(hard_negative, 1, rng)
        selected_negative += _sample_without_replacement(regular_negative, 1, rng)
        if len(selected_positive) < 2 or len(selected_negative) < 2:
            skipped += 1
            continue

        for current_frame, label in [
            *((frame, 1) for frame in selected_positive),
            *((frame, 0) for frame in selected_negative),
        ]:
            history = case.signals[
                point_index,
                current_frame - HISTORY_FRAMES + 1 : current_frame + 1,
            ]
            xyz = case.points[point_index]
            time_fraction = np.asarray([current_frame / (case.signals.shape[1] - 1)], dtype=np.float32)
            features.append(np.concatenate([history, xyz, time_fraction]).astype(np.float32))
            labels.append(label)

    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(y.size)
    counts = {"points": int(case.points.shape[0]), "examples": int(y.size), "skippedPoints": skipped}
    return x[order], y[order], counts


def concatenate_cases(cases: list[CaseArrays], seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    all_x = []
    all_y = []
    summaries = []
    for offset, case in enumerate(cases):
        x, y, summary = make_examples(case, seed + offset * 1009)
        all_x.append(x)
        all_y.append(y)
        summaries.append({"caseId": case.case_id, **summary})
    return np.concatenate(all_x), np.concatenate(all_y), summaries


def standardization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std

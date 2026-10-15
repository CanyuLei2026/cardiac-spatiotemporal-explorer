from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.data import activation_frames, load_mat_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cardiac MAT cases before selecting the five-case split.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("case_*.mat"))[: args.limit]
    if not paths:
        raise SystemExit(f"No case_*.mat files found in {args.data_dir}")

    print("case\tpoints\tframes\tfinite\tactivation frame q10/q50/q90")
    for path in paths:
        try:
            case = load_mat_case(path)
            activations = activation_frames(case.signals)
            quantiles = np.quantile(activations, [0.1, 0.5, 0.9]).astype(int)
            finite = bool(np.isfinite(case.points).all() and np.isfinite(case.signals).all())
            print(
                f"{case.case_id}\t{case.points.shape[0]}\t{case.signals.shape[1]}\t"
                f"{finite}\t{quantiles[0]}/{quantiles[1]}/{quantiles[2]}"
            )
        except Exception as error:  # Audit should continue and report every unusable case.
            print(f"{path.stem}\tERROR\t{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()

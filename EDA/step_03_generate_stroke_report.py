#!/usr/bin/env python3
"""EDA Step 3 — generate stroke_report.html from the complete raw dataset.

This profiles the in-memory DataFrame built from all 32 raw ``.npy`` files.
Minimal profiling mode is intentional: it calculates the guide's initial
dataset overview for every row while avoiding costly pairwise interactions on
more than four million timepoints. It does not sample or preprocess signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from data_profiling import ProfileReport

try:
    from EDA.step_02_join_all_raw_movements import load_all_raw_movements
except ModuleNotFoundError:  # Permit direct execution from inside EDA/.
    from step_02_join_all_raw_movements import load_all_raw_movements  # type: ignore[no-redef]


def generate_report(
    output: str | Path,
    frame: pd.DataFrame | None = None,
    raw_dir: str | Path | None = None,
) -> Path:
    """Build and write the full-dataset EDA overview; return its output path."""
    output_path = Path(output).resolve()
    if frame is not None and raw_dir is not None:
        raise ValueError("Pass either an existing frame or raw_dir, not both.")
    dataset_frame = frame if frame is not None else load_all_raw_movements(raw_dir)
    profile = ProfileReport(
        dataset_frame,
        title="REHAB Exercise Raw Dataset — All Movements",
        minimal=True,
        dataset={
            "description": (
                "All 4,616 raw REHAB exercise recordings joined at timepoint "
                "level. The target variable is movement_type; record_id is the "
                "independent observation unit. No signal preprocessing applied."
            ),
            "url": "https://doi.org/10.1038/s41597-026-07802-2",
        },
    )
    profile.to_file(output_path)
    return output_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "stroke_report.html",
        help="HTML destination (default: project-root stroke_report.html)",
    )
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    print(generate_report(args.output, raw_dir=args.raw_dir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EDA Step 1 — load untouched REHAB raw recordings into pandas DataFrames.

This module performs no signal preprocessing and writes no data files. It only
changes the in-memory layout from paired NumPy arrays to a tabular pandas
DataFrame so the raw observations can be inspected interactively.

Use from a notebook or Python session:

    from EDA.step_01_load_raw_dataframes import load_record, load_movement
    record = load_record(0, 0)
    movement = load_movement(0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError as error:  # Keep the dependency failure actionable.
    raise ImportError(
        "pandas is required to use this loader. Install it in the project environment with "
        "./.venv/bin/python -m pip install pandas"
    ) from error


MOVEMENTS = (
    "bobath_handshake",
    "bobath_flexion_extension",
    "bobath_forward_flexion_extension",
    "bobath_anterior_posterior_rotation",
    "elbow_flexion_wrist_compression",
    "wrist_flexion_extension",
    "finger_to_finger_training",
    "ball_gripping",
    "shoulder_internal_external_rotation",
    "breast_expansion",
    "flexion_pressure_rotation",
    "elbow_flexion_touch",
    "shoulder_touch_training",
    "ankle_extension_knee_rotation",
    "knee_flexion_extension",
    "hip_flexion_extension",
)

SENSOR_COLUMNS = (
    "pitch_1", "yaw_1", "roll_1", "pitch_2", "yaw_2", "roll_2",
    "finger_1", "finger_2", "finger_3", "finger_4", "finger_5", "wrist_pitch",
)


def _default_raw_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "REHAB" / "Rehab_exercise" / "d01_raw_data"


def _load_pair(movement_id: int, raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if movement_id not in range(len(MOVEMENTS)):
        raise ValueError(f"movement_id must be between 0 and {len(MOVEMENTS) - 1}")
    imu = np.load(raw_dir / f"{movement_id:03d}_1.npy", mmap_mode="r", allow_pickle=False)
    glove = np.load(raw_dir / f"{movement_id:03d}_2.npy", mmap_mode="r", allow_pickle=False)
    if imu.shape != glove.shape or imu.ndim != 3 or imu.shape[1:] != (880, 6):
        raise ValueError(f"Expected matching (n_records, 880, 6) arrays; received {imu.shape} and {glove.shape}")
    return imu, glove


def load_record(
    movement_id: int,
    sample_id: int,
    raw_dir: str | Path | None = None,
) -> "pd.DataFrame":
    """Return one raw recording as an 880-row, 17-column DataFrame.

    The 12 sensor columns retain their stored float64 values. The five metadata
    columns identify the source recording and timepoint.
    """
    raw_path = Path(raw_dir) if raw_dir is not None else _default_raw_dir()
    imu, glove = _load_pair(movement_id, raw_path)
    if sample_id not in range(imu.shape[0]):
        raise ValueError(f"sample_id must be between 0 and {imu.shape[0] - 1} for movement {movement_id:03d}")

    values = np.concatenate((imu[sample_id], glove[sample_id]), axis=1)
    frame = pd.DataFrame(values, columns=SENSOR_COLUMNS, copy=False)
    frame.insert(0, "time_s", np.arange(len(frame)) / 50)
    frame.insert(0, "timepoint", np.arange(len(frame), dtype=np.int16))
    frame.insert(0, "sample_id", sample_id)
    frame.insert(0, "movement_name", MOVEMENTS[movement_id])
    frame.insert(0, "movement_id", f"{movement_id:03d}")
    return frame


def load_movement(
    movement_id: int,
    raw_dir: str | Path | None = None,
) -> "pd.DataFrame":
    """Return all raw recordings of one movement as a timepoint-level DataFrame.

    This is a structural conversion only. For movement 000, for example, the
    returned table has 232 × 880 = 204,160 rows and 17 columns.
    """
    raw_path = Path(raw_dir) if raw_dir is not None else _default_raw_dir()
    imu, glove = _load_pair(movement_id, raw_path)
    records, timepoints, _ = imu.shape
    values = np.concatenate((imu, glove), axis=2).reshape(records * timepoints, len(SENSOR_COLUMNS))
    frame = pd.DataFrame(values, columns=SENSOR_COLUMNS, copy=False)
    frame.insert(0, "time_s", np.tile(np.arange(timepoints) / 50, records))
    frame.insert(0, "timepoint", np.tile(np.arange(timepoints, dtype=np.int16), records))
    frame.insert(0, "sample_id", np.repeat(np.arange(records, dtype=np.int32), timepoints))
    frame.insert(0, "movement_name", MOVEMENTS[movement_id])
    frame.insert(0, "movement_id", f"{movement_id:03d}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movement-id", type=int, required=True, choices=range(len(MOVEMENTS)))
    parser.add_argument("--sample-id", type=int, help="Load one recording instead of all records for the movement.")
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()

    frame = (
        load_record(args.movement_id, args.sample_id, args.raw_dir)
        if args.sample_id is not None
        else load_movement(args.movement_id, args.raw_dir)
    )
    print(frame.shape)
    print(frame.head())
    print(frame.dtypes)


if __name__ == "__main__":
    main()

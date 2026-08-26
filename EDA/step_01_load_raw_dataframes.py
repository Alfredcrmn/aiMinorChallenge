#!/usr/bin/env python3
"""EDA Step 1 — load processed REHAB recordings into pandas DataFrames.

The source arrays have already been filtered and centered by the dataset
authors. This module only changes their in-memory layout from paired NumPy
arrays to tabular pandas DataFrames; it writes no data files.

Use from a notebook or Python session:

    from EDA.step_01_load_raw_dataframes import load_record, load_movement
    record = load_record(0, 0)
    movement = load_movement(0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "REHAB" / "Rehab_exercise" / "d02_processed_data"


def _load_pair(movement_id: int, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if movement_id not in range(len(MOVEMENTS)):
        raise ValueError(f"movement_id must be between 0 and {len(MOVEMENTS) - 1}")
    imu = np.load(data_dir / f"{movement_id:03d}_1.npy", mmap_mode="r", allow_pickle=False)
    glove = np.load(data_dir / f"{movement_id:03d}_2.npy", mmap_mode="r", allow_pickle=False)
    if (
        imu.shape != glove.shape
        or imu.ndim != 3
        or not imu.shape[0]
        or imu.shape[1:] != (880, 6)
    ):
        raise ValueError(
            "Expected matching, non-empty (n_records, 880, 6) arrays; "
            f"received {imu.shape} and {glove.shape}"
        )
    return imu, glove


def _to_frame(
    movement_id: int,
    imu: np.ndarray,
    glove: np.ndarray,
    sample_ids: np.ndarray,
) -> pd.DataFrame:
    """Combine aligned device arrays and attach recording metadata."""
    records, timepoints, _ = imu.shape
    values = np.concatenate((imu, glove), axis=2).reshape(-1, len(SENSOR_COLUMNS))
    frame = pd.DataFrame(values, columns=SENSOR_COLUMNS, copy=False)
    frame.insert(0, "time_s", np.tile(np.arange(timepoints) / 50.0, records))
    frame.insert(0, "timepoint", np.tile(np.arange(timepoints, dtype=np.int16), records))
    frame.insert(0, "sample_id", np.repeat(sample_ids, timepoints))
    frame.insert(0, "movement_name", MOVEMENTS[movement_id])
    frame.insert(0, "movement_id", f"{movement_id:03d}")
    return frame


def load_record(
    movement_id: int,
    sample_id: int,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return one processed recording as an 880-row, 17-column DataFrame.

    The 12 sensor columns retain their stored float64 values. The five metadata
    columns identify the source recording and timepoint.
    """
    data_path = Path(data_dir) if data_dir is not None else _default_data_dir()
    imu, glove = _load_pair(movement_id, data_path)
    if sample_id not in range(imu.shape[0]):
        raise ValueError(
            f"sample_id must be between 0 and {imu.shape[0] - 1} "
            f"for movement {movement_id:03d}"
        )
    return _to_frame(
        movement_id,
        imu[sample_id : sample_id + 1],
        glove[sample_id : sample_id + 1],
        np.array([sample_id], dtype=np.int32),
    )


def load_movement(
    movement_id: int,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return all processed recordings of one movement as a timepoint-level DataFrame.

    This is a structural conversion only. For movement 000, for example, the
    returned table has 232 × 880 = 204,160 rows and 17 columns.
    """
    data_path = Path(data_dir) if data_dir is not None else _default_data_dir()
    imu, glove = _load_pair(movement_id, data_path)
    return _to_frame(movement_id, imu, glove, np.arange(imu.shape[0], dtype=np.int32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movement-id", type=int, required=True, choices=range(len(MOVEMENTS)))
    parser.add_argument("--sample-id", type=int, help="Load one recording instead of all records for the movement.")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    frame = (
        load_record(args.movement_id, args.sample_id, args.data_dir)
        if args.sample_id is not None
        else load_movement(args.movement_id, args.data_dir)
    )
    print(frame.shape)
    print(frame.head())
    print(frame.dtypes)


if __name__ == "__main__":
    main()

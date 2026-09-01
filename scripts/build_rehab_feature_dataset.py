#!/usr/bin/env python3
"""Build a tabular feature dataset from the processed REHAB exercise data.

Each original recording contains 880 timepoints and 12 sensor channels. This
script divides every recording into eight contiguous, non-overlapping windows
of 110 timepoints and extracts six features per channel. The train/test split
is stratified by movement and assigned at recording level to prevent leakage
between windows from the same recording.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np


MOVEMENTS = [
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
]

SENSOR_COLUMNS = [
    "pitch_1",
    "yaw_1",
    "roll_1",
    "pitch_2",
    "yaw_2",
    "roll_2",
    "finger_1",
    "finger_2",
    "finger_3",
    "finger_4",
    "finger_5",
    "wrist_pitch",
]

FEATURES = ("std", "median", "min", "max", "iqr", "mad_diff")
METADATA_COLUMNS = (
    "record_id",
    "source_sample_id",
    "movement_id",
    "movement_type",
    "window_id",
    "start_timepoint",
    "end_timepoint",
    "split",
)

TIMEPOINT_COUNT = 880
WINDOW_COUNT = 8
WINDOW_SIZE = TIMEPOINT_COUNT // WINDOW_COUNT
CHANNEL_COUNT_PER_FILE = 6


def extract_features(values: np.ndarray) -> dict[str, float]:
    """Extract the six agreed features from one one-dimensional signal."""
    if values.ndim != 1 or values.size != WINDOW_SIZE:
        raise ValueError(
            f"Expected a one-dimensional window of {WINDOW_SIZE} values; "
            f"received shape {values.shape}"
        )

    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    differences = np.diff(values)

    return {
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "iqr": float(q75 - q25),
        "mad_diff": float(np.mean(np.abs(differences))),
    }


def feature_columns() -> list[str]:
    """Return feature columns in their stable CSV order."""
    return [
        f"{sensor_name}_{feature_name}"
        for sensor_name in SENSOR_COLUMNS
        for feature_name in FEATURES
    ]


def select_test_samples(
    sample_count: int, movement_id: int, test_fraction: float, seed: int
) -> set[int]:
    """Select a reproducible, movement-stratified set of test recordings."""
    if sample_count < 2:
        raise ValueError("At least two samples per movement are required")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    sample_ids = list(range(sample_count))
    random.Random(seed + movement_id).shuffle(sample_ids)
    test_count = max(1, min(sample_count - 1, round(sample_count * test_fraction)))
    return set(sample_ids[:test_count])


def validate_source_arrays(
    imu: np.ndarray, glove: np.ndarray, movement_id: int
) -> None:
    """Validate the paired processed arrays before extracting features."""
    expected_tail = (TIMEPOINT_COUNT, CHANNEL_COUNT_PER_FILE)
    if imu.shape != glove.shape or imu.ndim != 3 or imu.shape[1:] != expected_tail:
        raise ValueError(
            f"Unexpected shapes for movement {movement_id:03d}: "
            f"imu={imu.shape}, glove={glove.shape}; expected (*, {expected_tail})"
        )
    if not np.issubdtype(imu.dtype, np.number) or not np.issubdtype(
        glove.dtype, np.number
    ):
        raise ValueError(f"Non-numeric source data for movement {movement_id:03d}")
    if not np.isfinite(imu).all() or not np.isfinite(glove).all():
        raise ValueError(f"Non-finite source data for movement {movement_id:03d}")


def format_feature(value: float) -> str:
    """Serialize features compactly while preserving useful numeric precision."""
    return format(value, ".12g")


def build_dataset(
    processed_dir: Path,
    output_path: Path,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[int, int]:
    """Generate the feature CSV and return its recording and row counts."""
    if TIMEPOINT_COUNT % WINDOW_COUNT != 0:
        raise ValueError("TIMEPOINT_COUNT must be divisible by WINDOW_COUNT")
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed-data directory not found: {processed_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    header = [*METADATA_COLUMNS, *feature_columns()]
    record_id = 0
    row_count = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, lineterminator="\n")
            writer.writerow(header)

            for movement_id, movement_name in enumerate(MOVEMENTS):
                imu = np.load(
                    processed_dir / f"{movement_id:03d}_1.npy", allow_pickle=False
                )
                glove = np.load(
                    processed_dir / f"{movement_id:03d}_2.npy", allow_pickle=False
                )
                validate_source_arrays(imu, glove, movement_id)
                test_samples = select_test_samples(
                    imu.shape[0], movement_id, test_fraction, seed
                )

                for source_sample_id in range(imu.shape[0]):
                    recording = np.concatenate(
                        [imu[source_sample_id], glove[source_sample_id]], axis=1
                    )
                    split = (
                        "test" if source_sample_id in test_samples else "train"
                    )

                    for window_id in range(WINDOW_COUNT):
                        start = window_id * WINDOW_SIZE
                        end_exclusive = start + WINDOW_SIZE
                        window = recording[start:end_exclusive]
                        feature_values: list[str] = []

                        for channel_index in range(len(SENSOR_COLUMNS)):
                            channel_features = extract_features(
                                window[:, channel_index]
                            )
                            feature_values.extend(
                                format_feature(channel_features[name])
                                for name in FEATURES
                            )

                        writer.writerow(
                            [
                                record_id,
                                source_sample_id,
                                movement_id,
                                movement_name,
                                window_id,
                                start,
                                end_exclusive - 1,
                                split,
                                *feature_values,
                            ]
                        )
                        row_count += 1

                    record_id += 1

            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return record_id, row_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_dataset_root = project_root / "REHAB" / "Rehab_exercise"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_dataset_root / "d02_processed_data",
        help="Directory containing the processed .npy files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dataset_root
        / "d03_feature_data"
        / "rehab_exercise_features.csv",
        help="Destination CSV path",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recording_count, row_count = build_dataset(
        processed_dir=args.processed_dir.resolve(),
        output_path=args.output.resolve(),
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"Dataset written to: {args.output.resolve()}")
    print(f"Recordings: {recording_count:,}")
    print(f"Windows: {row_count:,}")
    print(f"Feature columns: {len(feature_columns())}")


if __name__ == "__main__":
    main()

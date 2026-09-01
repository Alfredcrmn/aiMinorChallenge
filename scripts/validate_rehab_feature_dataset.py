#!/usr/bin/env python3
"""Validate the derived REHAB exercise feature dataset."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_rehab_feature_dataset import (
    METADATA_COLUMNS,
    MOVEMENTS,
    TIMEPOINT_COUNT,
    WINDOW_COUNT,
    WINDOW_SIZE,
    feature_columns,
)


def source_sample_counts(processed_dir: Path) -> list[int]:
    """Return source recording counts after validating all paired shapes."""
    counts: list[int] = []
    for movement_id in range(len(MOVEMENTS)):
        imu = np.load(
            processed_dir / f"{movement_id:03d}_1.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        glove = np.load(
            processed_dir / f"{movement_id:03d}_2.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        expected_tail = (TIMEPOINT_COUNT, 6)
        if imu.shape != glove.shape or imu.shape[1:] != expected_tail:
            raise ValueError(
                f"Unexpected source shapes for movement {movement_id:03d}: "
                f"imu={imu.shape}, glove={glove.shape}"
            )
        counts.append(imu.shape[0])
    return counts


def parse_int(row: dict[str, str], column: str, line_number: int) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid integer in column {column!r} at CSV line {line_number}"
        ) from error


def validate_dataset(
    csv_path: Path, processed_dir: Path, test_fraction: float = 0.20
) -> dict[str, object]:
    """Run structural, numeric, provenance, split, and leakage validations."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Feature dataset not found: {csv_path}")

    expected_header = [*METADATA_COLUMNS, *feature_columns()]
    expected_source_counts = source_sample_counts(processed_dir)
    expected_record_count = sum(expected_source_counts)
    expected_row_count = expected_record_count * WINDOW_COUNT

    seen_windows: set[tuple[int, int]] = set()
    record_windows: dict[int, set[int]] = defaultdict(set)
    record_splits: dict[int, set[str]] = defaultdict(set)
    record_metadata: dict[int, tuple[int, int, str]] = {}
    source_to_record: dict[tuple[int, int], int] = {}
    records_per_movement: Counter[int] = Counter()
    records_per_movement_split: Counter[tuple[int, str]] = Counter()
    rows_per_split: Counter[str] = Counter()
    row_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != expected_header:
            raise ValueError(
                "Unexpected CSV header. "
                f"Expected {len(expected_header)} columns; "
                f"received {len(reader.fieldnames or [])}"
            )

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            record_id = parse_int(row, "record_id", line_number)
            source_sample_id = parse_int(row, "source_sample_id", line_number)
            movement_id = parse_int(row, "movement_id", line_number)
            window_id = parse_int(row, "window_id", line_number)
            start = parse_int(row, "start_timepoint", line_number)
            end = parse_int(row, "end_timepoint", line_number)
            movement_type = row["movement_type"]
            split = row["split"]

            if not 0 <= movement_id < len(MOVEMENTS):
                raise ValueError(f"Invalid movement_id at CSV line {line_number}")
            if movement_type != MOVEMENTS[movement_id]:
                raise ValueError(f"Movement label mismatch at CSV line {line_number}")
            if not 0 <= source_sample_id < expected_source_counts[movement_id]:
                raise ValueError(f"Invalid source_sample_id at CSV line {line_number}")
            if not 0 <= window_id < WINDOW_COUNT:
                raise ValueError(f"Invalid window_id at CSV line {line_number}")
            if start != window_id * WINDOW_SIZE or end != start + WINDOW_SIZE - 1:
                raise ValueError(f"Incorrect window boundaries at CSV line {line_number}")
            if split not in {"train", "test"}:
                raise ValueError(f"Invalid split at CSV line {line_number}")

            key = (record_id, window_id)
            if key in seen_windows:
                raise ValueError(f"Duplicate record/window pair at CSV line {line_number}")
            seen_windows.add(key)
            record_windows[record_id].add(window_id)
            record_splits[record_id].add(split)
            rows_per_split[split] += 1

            metadata = (source_sample_id, movement_id, movement_type)
            previous_metadata = record_metadata.setdefault(record_id, metadata)
            if previous_metadata != metadata:
                raise ValueError(
                    f"Inconsistent provenance for record_id={record_id}"
                )

            source_key = (movement_id, source_sample_id)
            previous_record_id = source_to_record.setdefault(source_key, record_id)
            if previous_record_id != record_id:
                raise ValueError(
                    f"Source recording {source_key} is mapped to multiple record_ids"
                )

            for column in feature_columns():
                try:
                    value = float(row[column])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid feature {column!r} at CSV line {line_number}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite feature {column!r} at CSV line {line_number}"
                    )

    if row_count != expected_row_count:
        raise ValueError(f"Expected {expected_row_count} rows; received {row_count}")
    if len(record_metadata) != expected_record_count:
        raise ValueError(
            f"Expected {expected_record_count} records; received {len(record_metadata)}"
        )
    if set(record_metadata) != set(range(expected_record_count)):
        raise ValueError("record_id values are not contiguous from zero")
    if len(source_to_record) != expected_record_count:
        raise ValueError("One or more source recordings are missing or duplicated")

    expected_windows = set(range(WINDOW_COUNT))
    for record_id in record_metadata:
        if record_windows[record_id] != expected_windows:
            raise ValueError(f"Incomplete windows for record_id={record_id}")
        if len(record_splits[record_id]) != 1:
            raise ValueError(
                f"Train/test leakage detected for record_id={record_id}"
            )
        movement_id = record_metadata[record_id][1]
        split = next(iter(record_splits[record_id]))
        records_per_movement[movement_id] += 1
        records_per_movement_split[(movement_id, split)] += 1

    for movement_id, expected_count in enumerate(expected_source_counts):
        if records_per_movement[movement_id] != expected_count:
            raise ValueError(
                f"Movement {movement_id:03d}: expected {expected_count} records; "
                f"received {records_per_movement[movement_id]}"
            )
        expected_test_count = max(
            1, min(expected_count - 1, round(expected_count * test_fraction))
        )
        if records_per_movement_split[(movement_id, "test")] != expected_test_count:
            raise ValueError(
                f"Movement {movement_id:03d}: expected {expected_test_count} test "
                f"records; received "
                f"{records_per_movement_split[(movement_id, 'test')]}"
            )

    return {
        "rows": row_count,
        "records": len(record_metadata),
        "features": len(feature_columns()),
        "train_rows": rows_per_split["train"],
        "test_rows": rows_per_split["test"],
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_dataset_root = project_root / "REHAB" / "Rehab_exercise"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_dataset_root
        / "d03_feature_data"
        / "rehab_exercise_features.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_dataset_root / "d02_processed_data",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_dataset(
        args.csv.resolve(), args.processed_dir.resolve(), args.test_fraction
    )
    print("Validation passed")
    print(f"Rows: {summary['rows']:,}")
    print(f"Records: {summary['records']:,}")
    print(f"Feature columns: {summary['features']}")
    print(f"Train rows: {summary['train_rows']:,}")
    print(f"Test rows: {summary['test_rows']:,}")
    print("Record-level train/test leakage: 0")


if __name__ == "__main__":
    main()

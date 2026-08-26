#!/usr/bin/env python3
"""EDA Step 2 — join every raw REHAB exercise array in one DataFrame.

The returned table contains all 4,616 recordings and all 880 stored timepoints
per recording. No signal values are filtered, normalized, imputed, resampled,
or aggregated, and no CSV or other intermediate dataset is written.

The table is timepoint-level so it can be explored with pandas. ``record_id``
identifies the 4,616 independent movement examples; future train/test splits
must group by this column rather than treating timepoints as independent.

Notebook usage:

    from EDA.step_02_join_all_raw_movements import load_all_raw_movements
    df = load_all_raw_movements()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from EDA.step_01_load_raw_dataframes import (
        MOVEMENTS,
        SENSOR_COLUMNS,
        _default_raw_dir,
        _load_pair,
    )
except ModuleNotFoundError:  # Permit direct execution from inside EDA/.
    from step_01_load_raw_dataframes import (  # type: ignore[no-redef]
        MOVEMENTS,
        SENSOR_COLUMNS,
        _default_raw_dir,
        _load_pair,
    )


TARGET_COLUMN = "movement_type"
METADATA_COLUMNS = (
    TARGET_COLUMN,
    "record_id",
    "sample_id",
    "timepoint",
    "time_s",
)


def load_all_raw_movements(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Return all paired ``.npy`` files as one raw timepoint-level DataFrame.

    ``movement_type`` is the categorical prediction target. ``sample_id`` is
    local to a movement class, while ``record_id`` is unique across the entire
    dataset. The 12 sensor columns contain the original stored float64 values.
    """
    raw_path = Path(raw_dir) if raw_dir is not None else _default_raw_dir()
    frames: list[pd.DataFrame] = []
    next_record_id = 0

    for movement_id, movement_type in enumerate(MOVEMENTS):
        imu, glove = _load_pair(movement_id, raw_path)
        records, timepoints, _ = imu.shape

        # Pair the two six-channel devices along the channel axis only. This
        # changes layout, not values, and keeps each recording/timepoint aligned.
        values = np.concatenate((imu, glove), axis=2).reshape(
            records * timepoints,
            len(SENSOR_COLUMNS),
        )
        frame = pd.DataFrame(values, columns=SENSOR_COLUMNS, copy=False)
        frame.insert(0, "time_s", np.tile(np.arange(timepoints) / 50.0, records))
        frame.insert(0, "timepoint", np.tile(np.arange(timepoints, dtype=np.int16), records))
        frame.insert(0, "sample_id", np.repeat(np.arange(records, dtype=np.int16), timepoints))
        frame.insert(
            0,
            "record_id",
            np.repeat(
                np.arange(next_record_id, next_record_id + records, dtype=np.int32),
                timepoints,
            ),
        )
        frame.insert(0, TARGET_COLUMN, movement_type)
        frames.append(frame)
        next_record_id += records

    combined = pd.concat(frames, ignore_index=True, copy=False)
    combined[TARGET_COLUMN] = pd.Categorical(
        combined[TARGET_COLUMN],
        categories=MOVEMENTS,
        ordered=False,
    )
    combined.attrs.update(
        {
            "target_column": TARGET_COLUMN,
            "observation_unit": "record_id",
            "sampling_frequency_hz": 50,
            "signal_preprocessing": "none",
        }
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()

    frame = load_all_raw_movements(args.raw_dir)
    print(frame.shape)
    print(frame.head())
    print("recordings:", frame["record_id"].nunique())
    print("target:", TARGET_COLUMN)
    print(frame.groupby(TARGET_COLUMN, observed=True)["record_id"].nunique())


if __name__ == "__main__":
    main()

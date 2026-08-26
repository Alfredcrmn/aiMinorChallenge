#!/usr/bin/env python3
"""EDA Step 2 — join every processed REHAB exercise array in one DataFrame.

The returned table contains all 4,616 recordings and all 880 stored timepoints
per recording. This step does not alter the already-processed signal values and
does not write an intermediate dataset.

The table is timepoint-level so it can be explored with pandas. ``record_id``
identifies the 4,616 independent movement examples; future train/test splits
must group by this column rather than treating timepoints as independent.

Notebook usage:

    from EDA.step_02_join_all_raw_movements import load_all_movements
    df = load_all_movements()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from EDA.step_01_load_raw_dataframes import (
        MOVEMENTS,
        _default_data_dir,
        load_movement,
    )
except ModuleNotFoundError:  # Permit direct execution from inside EDA/.
    from step_01_load_raw_dataframes import (  # type: ignore[no-redef]
        MOVEMENTS,
        _default_data_dir,
        load_movement,
    )


TARGET_COLUMN = "movement_type"


def load_all_movements(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return all paired processed ``.npy`` files as one timepoint-level DataFrame.

    ``movement_type`` is the categorical prediction target. ``sample_id`` is
    local to a movement class, while ``record_id`` is unique across the entire
    dataset. The 12 sensor columns contain the stored processed float64 values.
    """
    data_path = Path(data_dir) if data_dir is not None else _default_data_dir()
    frames: list[pd.DataFrame] = []
    next_record_id = 0

    for movement_id in range(len(MOVEMENTS)):
        frame = load_movement(movement_id, data_path)
        records = int(frame["sample_id"].iat[-1]) + 1
        frame.drop(columns="movement_id", inplace=True)
        frame.rename(columns={"movement_name": TARGET_COLUMN}, inplace=True)
        frame.insert(1, "record_id", (frame["sample_id"] + next_record_id).astype(np.int32))
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
            "signal_preprocessing": "dataset-provided processed arrays",
        }
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    frame = load_all_movements(args.data_dir)
    print(frame.shape)
    print(frame.head())
    print("recordings:", frame["record_id"].nunique())
    print("target:", TARGET_COLUMN)
    print(frame.groupby(TARGET_COLUMN, observed=True)["record_id"].nunique())


if __name__ == "__main__":
    main()

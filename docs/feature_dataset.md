# REHAB exercise feature dataset

This derived dataset converts the processed rehabilitation exercise signals
into a flat table suitable for classical machine-learning algorithms.

The source is the processed `Rehab_exercise` subset of the REHAB dataset
described in [A wearable sensor-based kinematic dataset collected under
standardized rehabilitation tasks from 120 post-stroke
patients](https://doi.org/10.1038/s41597-026-07802-2).

## Observation unit

Each original recording contains 880 timepoints from 12 sensor channels. A
recording is divided chronologically into eight contiguous, non-overlapping
windows of 110 timepoints. At the dataset's approximate sampling frequency of
50 Hz, each window represents approximately 2.2 seconds.

The timepoints are never shuffled. Randomization is used only to assign complete
recordings to the training or test split.

## Features

Six features are calculated separately for each of the 12 channels, resulting
in 72 numeric feature columns per window:

- `std`: population standard deviation.
- `median`: median.
- `min`: minimum value.
- `max`: maximum value.
- `iqr`: interquartile range (`75th percentile - 25th percentile`).
- `mad_diff`: mean absolute difference between consecutive timepoints.

Feature names follow the pattern `<sensor>_<feature>`, for example
`pitch_1_std` and `wrist_pitch_mad_diff`.

## Metadata and provenance

Every CSV row contains these metadata columns before the 72 features:

| Column | Description |
| --- | --- |
| `record_id` | Globally unique identifier for the original recording. |
| `source_sample_id` | Index of the recording inside its source `.npy` pair. |
| `movement_id` | Numeric movement class from 0 through 15. |
| `movement_type` | Human-readable movement label. |
| `window_id` | Chronological window index from 0 through 7. |
| `start_timepoint` | Inclusive initial timepoint in the original recording. |
| `end_timepoint` | Inclusive final timepoint in the original recording. |
| `split` | Either `train` or `test`. |

The default output is:

```text
REHAB/Rehab_exercise/d03_feature_data/rehab_exercise_features.csv
```

With the current source files, the result contains 4,616 original recordings,
36,928 window rows, 8 metadata columns, and 72 feature columns.

## Train/test split

The split is stratified by movement, uses 80% of recordings for training and
20% for testing, and is reproducible with random seed 42. Split assignment is
performed before window extraction. Consequently, all eight windows belonging
to a `record_id` are always assigned to the same split, preventing leakage
between windows from the same recording.

The default split contains 3,693 training recordings (29,544 windows) and 923
test recordings (7,384 windows).

The source files do not expose a patient identifier for these exercise samples.
The split therefore guarantees independence by recording, but patient-level
independence cannot be verified from the available files.

## Reproduce and validate

From the repository root, run:

```bash
python scripts/build_rehab_feature_dataset.py
python scripts/validate_rehab_feature_dataset.py
python -m unittest tests/test_feature_dataset.py
```

The validator checks the schema, row and recording counts, source provenance,
window boundaries, numeric finiteness, class counts, completeness of all eight
windows, and absence of train/test leakage by `record_id`.

This shared data-preparation stage does not contain a machine-learning model or
evaluation metrics. Those components are intentionally left for the individual
part of the activity.

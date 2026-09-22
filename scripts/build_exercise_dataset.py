"""Create data/rehab_exercise.csv from the processed REHAB exercise files."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REHAB/Rehab_exercise/d02_processed_data"
OUTPUT = ROOT / "data/rehab_exercise.csv"
CHANNELS = ["pitch1", "yaw1", "roll1", "pitch2", "yaw2", "roll2",
            "f1", "f2", "f3", "f4", "f5", "pitch3"]
FEATURES = ["mean", "std", "rms", "range", "mean_abs_difference", "slope"]


def features(data):
    # Eight non-overlapping windows of 110 points; sampling rate is 50 Hz.
    if data.ndim != 3 or data.shape[1:] != (880, 6):
        raise ValueError(f"Expected (samples, 880, 6), got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Source data contains NaN or infinite values")
    w = np.asarray(data, dtype=np.float64).reshape(-1, 8, 110, 6)
    t = np.arange(110, dtype=np.float64) / 50
    t -= t.mean()
    return np.stack([
        w.mean(axis=2),
        w.std(axis=2, ddof=0),
        np.sqrt(np.mean(w ** 2, axis=2)),
        np.ptp(w, axis=2),
        np.abs(np.diff(w, axis=2)).mean(axis=2),
        np.einsum("nwtc,t->nwc", w, t) / np.dot(t, t),
    ], axis=-1)


def main():
    columns = [f"{channel}_{feature}" for channel in CHANNELS for feature in FEATURES]
    frames = []
    audit = []
    next_repetition_id = 1
    for exercise in range(16):
        if exercise == 14:  # Exclude both 014_1.npy and 014_2.npy before loading.
            continue
        arrays = [np.load(SOURCE / f"{exercise:03d}_{sensor}.npy",
                          allow_pickle=False) for sensor in (1, 2)]
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(f"Sensor sample counts differ for exercise {exercise}")
        if arrays[0].ndim != 3 or arrays[0].shape[1:] != (880, 6):
            raise ValueError(f"Unexpected source shape for exercise {exercise}")
        # Remove the entire paired recording if either sensor contains NaN/inf.
        # Preserve original sample indices; never interpolate unknown measurements.
        valid = np.isfinite(arrays[0]).all(axis=(1, 2)) & np.isfinite(arrays[1]).all(axis=(1, 2))
        seen, source_indices = set(), []
        for i in np.flatnonzero(valid):
            digest = hashlib.sha256(arrays[0][i].tobytes() + arrays[1][i].tobytes()).digest()
            if digest not in seen:
                seen.add(digest)
                source_indices.append(i)
        # Deduplicate only full, exact paired recordings within the same exercise.
        audit.append({"exercise": exercise, "input_recordings": len(valid),
                      "removed_nonfinite": int((~valid).sum()),
                      "removed_duplicates": int(valid.sum()) - len(source_indices),
                      "kept_recordings": len(source_indices)})
        nonempty = [i for i in source_indices if np.any(arrays[0][i]) or np.any(arrays[1][i])]
        audit[-1]["removed_all_zero"] = len(source_indices) - len(nonempty)
        source_indices = nonempty
        audit[-1]["kept_recordings"] = len(source_indices)
        if not source_indices:
            raise ValueError(f"No valid recordings for exercise {exercise}")
        sensors = [features(array[source_indices]) for array in arrays]
        # Matching sample indices in _1 and _2 are treated as one recording.
        x = np.concatenate(sensors, axis=2)
        frame = pd.DataFrame(x.reshape(-1, 72), columns=columns)
        frame.insert(0, "window", np.tile(np.arange(1, 9), len(x)))
        frame.insert(0, "sample_id", np.repeat(
            [f"{exercise:03d}:{i:04d}" for i in source_indices], 8))
        # Identify each source recording across its eight windows, not detected motion cycles.
        frame.insert(1, "repetition_id", np.repeat(
            np.arange(next_repetition_id, next_repetition_id + len(x)), 8))
        next_repetition_id += len(x)
        frame["exercise"] = exercise
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT, index=False, float_format="%.10g")
    pd.DataFrame(audit).to_csv(OUTPUT.parent / "cleaning_summary.csv", index=False)
    print(f"Saved {len(dataset):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

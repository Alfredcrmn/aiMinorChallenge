"""Create data/rehab_exercise.csv from the processed REHAB exercise files."""

from pathlib import Path

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
    next_repetition_id = 1
    for exercise in range(16):
        if exercise == 14:  # Exclude both 014_1.npy and 014_2.npy before loading.
            continue
        sensors = [features(np.load(SOURCE / f"{exercise:03d}_{sensor}.npy",
                                    allow_pickle=False)) for sensor in (1, 2)]
        if sensors[0].shape != sensors[1].shape:
            raise ValueError(f"Sensor sample counts differ for exercise {exercise}")
        # Matching sample indices in _1 and _2 are treated as one recording.
        x = np.concatenate(sensors, axis=2)
        frame = pd.DataFrame(x.reshape(-1, 72), columns=columns)
        frame.insert(0, "window", np.tile(np.arange(1, 9), len(x)))
        frame.insert(0, "sample_id", np.repeat(
            [f"{exercise:03d}:{i:04d}" for i in range(len(x))], 8))
        # Identify each source recording across its eight windows, not detected motion cycles.
        frame.insert(1, "repetition_id", np.repeat(
            np.arange(next_repetition_id, next_repetition_id + len(x)), 8))
        next_repetition_id += len(x)
        frame["exercise"] = exercise
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT, index=False, float_format="%.10g")
    print(f"Saved {len(dataset):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconstruct the corrupted processed REHAB exercise array 014_1.npy."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


EXPECTED_SHAPE = (359, 880, 6)
WINDOW_SIZE = 10


def preprocess(raw: np.ndarray) -> np.ndarray:
    """Apply the exercise preprocessing documented by the REHAB paper."""
    kernel = np.ones(WINDOW_SIZE, dtype=np.float64) / WINDOW_SIZE
    processed = np.empty_like(raw, dtype=np.float64)

    for sample_index in range(raw.shape[0]):
        for channel_index in range(raw.shape[2]):
            processed[sample_index, :, channel_index] = np.convolve(
                raw[sample_index, :, channel_index], kernel, mode="same"
            )

    processed -= processed.mean(axis=1, keepdims=True)
    return processed


def validate_raw(raw: np.ndarray) -> None:
    if raw.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected raw shape: {raw.shape}; expected {EXPECTED_SHAPE}")
    if raw.dtype != np.float64:
        raise ValueError(f"Unexpected raw dtype: {raw.dtype}; expected float64")
    if not np.isfinite(raw).all():
        raise ValueError("Raw data contain NaN or infinite values")
    if np.any(np.all(raw == 0, axis=(1, 2))):
        raise ValueError("Raw data contain one or more entirely zero samples")


def validate_processed(processed: np.ndarray) -> None:
    if processed.shape != EXPECTED_SHAPE or processed.dtype != np.float64:
        raise ValueError("Reconstructed data have an unexpected shape or dtype")
    if not np.isfinite(processed).all():
        raise ValueError("Reconstructed data contain NaN or infinite values")
    max_abs_mean = float(np.max(np.abs(processed.mean(axis=1))))
    if max_abs_mean > 1e-10:
        raise ValueError(f"Zero-mean validation failed: maximum mean={max_abs_mean}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "REHAB" / "Rehab_exercise",
    )
    args = parser.parse_args()

    raw_path = args.dataset_root / "d01_raw_data" / "014_1.npy"
    output_path = args.dataset_root / "d02_processed_data" / "014_1.npy"
    backup_path = output_path.with_suffix(".npy.corrupted.bak")
    temporary_path = output_path.with_suffix(".npy.reconstructed.tmp")

    raw = np.load(raw_path, allow_pickle=False)
    validate_raw(raw)
    processed = preprocess(raw)
    validate_processed(processed)

    if backup_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup_path}")
    shutil.copy2(output_path, backup_path)

    with temporary_path.open("wb") as temporary_file:
        np.save(temporary_file, processed, allow_pickle=False)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())

    reloaded = np.load(temporary_path, allow_pickle=False)
    if not np.array_equal(reloaded, processed):
        raise ValueError("Temporary output failed exact save/reload verification")

    os.replace(temporary_path, output_path)
    print(f"Reconstructed: {output_path}")
    print(f"Corrupted backup: {backup_path}")
    print(f"Shape: {processed.shape}; dtype: {processed.dtype}")
    print(f"Maximum absolute channel mean: {np.max(np.abs(processed.mean(axis=1))):.3e}")


if __name__ == "__main__":
    main()

"""Unit tests for the REHAB feature-dataset construction."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_rehab_feature_dataset import (  # noqa: E402
    FEATURES,
    SENSOR_COLUMNS,
    WINDOW_SIZE,
    extract_features,
    feature_columns,
    select_test_samples,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_features(self) -> None:
        values = np.arange(WINDOW_SIZE, dtype=float)
        result = extract_features(values)

        self.assertEqual(set(result), set(FEATURES))
        self.assertTrue(math.isclose(result["std"], float(np.std(values))))
        self.assertTrue(math.isclose(result["median"], 54.5))
        self.assertEqual(result["min"], 0.0)
        self.assertEqual(result["max"], 109.0)
        self.assertTrue(math.isclose(result["iqr"], 54.5))
        self.assertTrue(math.isclose(result["mad_diff"], 1.0))

    def test_feature_column_count_and_order(self) -> None:
        columns = feature_columns()
        self.assertEqual(len(columns), len(SENSOR_COLUMNS) * len(FEATURES))
        self.assertEqual(columns[0], "pitch_1_std")
        self.assertEqual(columns[-1], "wrist_pitch_mad_diff")

    def test_stratified_selection_is_reproducible(self) -> None:
        first = select_test_samples(100, movement_id=3, test_fraction=0.20, seed=42)
        second = select_test_samples(100, movement_id=3, test_fraction=0.20, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertTrue(all(0 <= sample_id < 100 for sample_id in first))

    def test_different_movements_use_different_randomizations(self) -> None:
        movement_zero = select_test_samples(
            100, movement_id=0, test_fraction=0.20, seed=42
        )
        movement_one = select_test_samples(
            100, movement_id=1, test_fraction=0.20, seed=42
        )

        self.assertNotEqual(movement_zero, movement_one)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nearest_centroid_from_scratch import (  # noqa: E402
    EXPECTED_CLASS_COUNT,
    EXPECTED_FEATURE_COUNT,
    METADATA_COLUMNS,
    load_dataset,
)


class DatasetLoadingTests(unittest.TestCase):
    def write_dataset(self, path: Path, invalid_feature: str | None = None) -> None:
        feature_names = [
            f"feature_{index}" for index in range(EXPECTED_FEATURE_COUNT)
        ]
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, lineterminator="\n")
            writer.writerow([*METADATA_COLUMNS, *feature_names])

            for class_id in range(EXPECTED_CLASS_COUNT):
                for split, record_offset in (("train", 0), ("test", 100)):
                    features = [
                        str(class_id + index / 100)
                        for index in range(EXPECTED_FEATURE_COUNT)
                    ]
                    if invalid_feature is not None and class_id == 0 and split == "train":
                        features[0] = invalid_feature
                    writer.writerow(
                        [
                            class_id + record_offset,
                            class_id,
                            class_id,
                            f"movement_{class_id}",
                            0,
                            0,
                            109,
                            split,
                            *features,
                        ]
                    )

    def test_loads_and_separates_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "features.csv"
            self.write_dataset(csv_path)

            dataset = load_dataset(csv_path)

        self.assertEqual(len(dataset.feature_names), EXPECTED_FEATURE_COUNT)
        self.assertEqual(len(dataset.class_names), EXPECTED_CLASS_COUNT)
        self.assertEqual(len(dataset.train), EXPECTED_CLASS_COUNT)
        self.assertEqual(len(dataset.test), EXPECTED_CLASS_COUNT)
        self.assertEqual(dataset.train[0].label, 0)
        self.assertEqual(len(dataset.train[0].features), EXPECTED_FEATURE_COUNT)

    def test_rejects_non_numeric_feature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "features.csv"
            self.write_dataset(csv_path, invalid_feature="not-a-number")

            with self.assertRaisesRegex(ValueError, "feature_0"):
                load_dataset(csv_path)

    def test_rejects_non_finite_feature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "features.csv"
            self.write_dataset(csv_path, invalid_feature="nan")

            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_dataset(csv_path)


if __name__ == "__main__":
    unittest.main()

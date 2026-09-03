import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nearest_centroid_from_scratch import METADATA_COLUMNS, EXPECTED_FEATURE_COUNT


class CentroidExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.csv_path = self.directory / "features.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                *METADATA_COLUMNS,
                *(f"feature_{i}" for i in range(EXPECTED_FEATURE_COUNT)),
            ])
            for label in range(16):
                for split_index, split in enumerate(("train", "test")):
                    # Last test window is deliberately near class 0, not class 15.
                    value = 0.1 if label == 15 and split == "test" else (
                        label * 10.0 + split_index * 0.1
                    )
                    writer.writerow([
                        label * 2 + split_index, label, label, f"class_{label}",
                        0, 0, 109, split,
                        *([value] * EXPECTED_FEATURE_COUNT),
                    ])

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-S", str(PROJECT_ROOT / "nearest_centroid_from_scratch.py"),
             "--csv", str(self.csv_path), *arguments],
            cwd=self.directory, capture_output=True, text=True, timeout=30,
        )

    def test_cli_executes_requested_examples_and_centroid_preview(self) -> None:
        result = self.run_cli("--examples", "3", "--show-centroids")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Centroids calculated: 16", result.stdout)
        self.assertIn("Test predictions shown: 3", result.stdout)
        self.assertIn("Preview coordinates:", result.stdout)
        rows = [line for line in result.stdout.splitlines() if line.startswith("record_id=")]
        self.assertEqual(len(rows), 3)
        for label, row in enumerate(rows):
            self.assertIn(f"actual={label:02d}, predicted={label:02d}", row)

    def test_zero_examples_and_default_compact_output(self) -> None:
        result = self.run_cli("--examples", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Test predictions shown: 0", result.stdout)
        self.assertNotIn("Preview coordinates:", result.stdout)

    def test_negative_examples_rejected(self) -> None:
        result = self.run_cli("--examples", "-1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative integer", result.stderr)

    def test_metrics_use_all_test_windows_regardless_of_example_limit(self) -> None:
        for limit in (0, 3, 100):
            with self.subTest(limit=limit):
                result = self.run_cli("--examples", str(limit))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Accuracy: 93.75% (test windows)", result.stdout)
                self.assertIn("Precision (macro): 90.62%", result.stdout)
                self.assertIn("Recall (macro): 93.75%", result.stdout)
                self.assertIn("F1 score (macro): 91.67%", result.stdout)
                self.assertIn("Correct predictions: 15 / 16", result.stdout)

    def test_missing_csv_has_clear_error(self) -> None:
        result = self.run_cli("--csv", str(self.directory / "missing.csv"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dataset not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

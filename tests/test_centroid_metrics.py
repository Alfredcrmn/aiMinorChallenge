import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nearest_centroid_from_scratch import calculate_accuracy, calculate_classification_metrics


class AccuracyTests(unittest.TestCase):
    def test_known_fraction(self) -> None:
        self.assertEqual(calculate_accuracy((0, 1, 2, 2), (0, 1, 0, 2)), 0.75)

    def test_all_correct_or_all_wrong(self) -> None:
        self.assertEqual(calculate_accuracy((0, 1), (0, 1)), 1.0)
        self.assertEqual(calculate_accuracy((0, 1), (1, 0)), 0.0)

    def test_empty_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_accuracy((), ())

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_accuracy((0, 1), (0,))


class ClassificationMetricsTests(unittest.TestCase):
    def test_known_multiclass_scores(self) -> None:
        result = calculate_classification_metrics((0, 0, 1, 1, 2), (0, 1, 1, 2, 2), 3)
        self.assertAlmostEqual(result.accuracy, 3 / 5)
        self.assertAlmostEqual(result.precision_macro, 2 / 3)
        self.assertAlmostEqual(result.recall_macro, 2 / 3)
        self.assertAlmostEqual(result.f1_macro, 11 / 18)
        # Averaging F1 by class must not be replaced by F1(macro P, macro R).
        self.assertNotAlmostEqual(result.f1_macro, 2 / 3)

    def test_zero_denominators_and_absent_class(self) -> None:
        result = calculate_classification_metrics((0, 0), (0, 1), 3)
        self.assertEqual(result.accuracy, 0.5)
        self.assertAlmostEqual(result.precision_macro, 1 / 3)
        self.assertAlmostEqual(result.recall_macro, 1 / 6)
        self.assertAlmostEqual(result.f1_macro, 2 / 9)

    def test_perfect_and_completely_wrong_predictions(self) -> None:
        for predicted, expected in (((0, 1, 2), 1.0), ((1, 2, 0), 0.0)):
            result = calculate_classification_metrics((0, 1, 2), predicted, 3)
            self.assertEqual(result.accuracy, expected)
            self.assertEqual(result.precision_macro, expected)
            self.assertEqual(result.recall_macro, expected)
            self.assertEqual(result.f1_macro, expected)

    def test_invalid_inputs(self) -> None:
        for actual, predicted, count in (
            ((), (), 3), ((0, 1), (0,), 3), ((0,), (0,), 0),
            ((-1,), (0,), 3), ((0,), (3,), 3), ((3,), (0,), 3),
        ):
            with self.subTest(actual=actual, predicted=predicted), self.assertRaises(ValueError):
                calculate_classification_metrics(actual, predicted, count)


if __name__ == "__main__":
    unittest.main()

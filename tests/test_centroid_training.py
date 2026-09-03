import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nearest_centroid_from_scratch import (
    Dataset,
    Sample,
    calculate_centroids,
    fit_standardizer,
    standardize_dataset,
    transform_features,
)


class CentroidTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset(
            feature_names=("a", "b"),
            class_names=("zero", "one", "two"),
            train=(
                Sample((-3.0, -1.0), 0, 0, 0),
                Sample((-1.0, 1.0), 0, 1, 0),
                Sample((2.0, 4.0), 1, 2, 0),
                Sample((3.0, 5.0), 1, 3, 0),
                Sample((4.0, 6.0), 1, 4, 0),
                Sample((10.0, 8.0), 2, 5, 0),
            ),
            test=(Sample((1e6, -1e6), 0, 9, 0),),
        )

    def test_classwise_means_counts_and_order(self) -> None:
        centroids = calculate_centroids(self.dataset)
        self.assertEqual([c.class_id for c in centroids], [0, 1, 2])
        self.assertEqual([c.class_name for c in centroids], ["zero", "one", "two"])
        self.assertEqual([c.sample_count for c in centroids], [2, 3, 1])
        self.assertEqual(
            [c.features for c in centroids], [(-2.0, 0.0), (3.0, 5.0), (10.0, 8.0)]
        )

    def test_test_rows_and_training_order_do_not_affect_centroids(self) -> None:
        expected = calculate_centroids(self.dataset)
        altered = replace(self.dataset, train=self.dataset.train[::-1], test=())
        self.assertEqual(calculate_centroids(altered), expected)

    def test_pipeline_equals_transformed_raw_class_means(self) -> None:
        scaler = fit_standardizer(self.dataset)
        standardized = standardize_dataset(self.dataset, scaler)
        actual = calculate_centroids(standardized)
        for centroid, raw_mean in zip(actual, ((-2.0, 0.0), (3.0, 5.0), (10.0, 8.0))):
            expected = transform_features(raw_mean, scaler)
            for value, reference in zip(centroid.features, expected):
                self.assertAlmostEqual(value, reference)
        self.assertEqual(self.dataset.train[0].features, (-3.0, -1.0))

    def test_rejects_missing_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "Class 2"):
            calculate_centroids(replace(self.dataset, train=self.dataset.train[:-1]))

    def test_rejects_invalid_training_data(self) -> None:
        cases = [
            replace(self.dataset, train=()),
            replace(self.dataset, feature_names=()),
            replace(self.dataset, class_names=()),
        ]
        invalid_samples = [
            Sample((1.0, 2.0), -1, 0, 0),
            Sample((1.0, 2.0), 3, 0, 0),
            Sample((1.0,), 0, 0, 0),
            Sample((math.nan, 2.0), 0, 0, 0),
            Sample((math.inf, 2.0), 0, 0, 0),
        ]
        cases += [replace(self.dataset, train=(sample,)) for sample in invalid_samples]
        for dataset in cases:
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                calculate_centroids(dataset)


if __name__ == "__main__":
    unittest.main()

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nearest_centroid_from_scratch import (
    Dataset,
    Sample,
    fit_standardizer,
    standardize_dataset,
    transform_features,
)


class StandardizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset(
            feature_names=("varying", "constant"),
            class_names=("zero", "one"),
            train=(Sample((1.0, 5.0), 0, 0, 0), Sample((3.0, 5.0), 1, 1, 7)),
            test=(Sample((100.0, 8.0), 1, 2, 3),),
        )

    def test_population_statistics_and_constant_divisor(self) -> None:
        scaler = fit_standardizer(self.dataset)
        self.assertEqual(scaler.means, (2.0, 5.0))
        self.assertEqual(scaler.scales, (1.0, 1.0))

    def test_transform_preserves_metadata_and_original_values(self) -> None:
        scaler = fit_standardizer(self.dataset)
        result = standardize_dataset(self.dataset, scaler)
        self.assertEqual(result.train[0].features, (-1.0, 0.0))
        self.assertEqual(result.train[1].features, (1.0, 0.0))
        self.assertEqual(result.test[0].features, (98.0, 3.0))
        self.assertEqual(self.dataset.train[0].features, (1.0, 5.0))
        self.assertEqual(result.class_names, self.dataset.class_names)
        self.assertEqual(result.feature_names, self.dataset.feature_names)
        for original, transformed in zip(
            self.dataset.train + self.dataset.test, result.train + result.test
        ):
            self.assertEqual(
                (original.label, original.record_id, original.window_id),
                (transformed.label, transformed.record_id, transformed.window_id),
            )

    def test_test_values_and_class_labels_do_not_affect_fit(self) -> None:
        altered = replace(
            self.dataset,
            train=tuple(replace(s, label=0) for s in self.dataset.train),
            test=(Sample((-1e10, 1e10), 0, 9, 0),),
        )
        self.assertEqual(fit_standardizer(self.dataset), fit_standardizer(altered))

    def test_training_mean_zero_and_population_std_one(self) -> None:
        dataset = replace(
            self.dataset,
            train=tuple(Sample((float(i), 5.0), 0, i, 0) for i in range(10)),
        )
        result = standardize_dataset(dataset, fit_standardizer(dataset))
        values = [s.features[0] for s in result.train]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(variance, 1.0)

    def test_single_training_sample(self) -> None:
        dataset = replace(self.dataset, train=self.dataset.train[:1])
        scaler = fit_standardizer(dataset)
        self.assertEqual(scaler.means, (1.0, 5.0))
        self.assertEqual(scaler.scales, (1.0, 1.0))

    def test_empty_or_invalid_training_data(self) -> None:
        invalid_sets = [(), (Sample((1.0,), 0, 0, 0),)]
        invalid_sets += [
            (Sample((value, 1.0), 0, 0, 0),)
            for value in (math.nan, math.inf, -math.inf)
        ]
        for samples in invalid_sets:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                fit_standardizer(replace(self.dataset, train=samples))

    def test_rejects_invalid_transform_inputs(self) -> None:
        scaler = fit_standardizer(self.dataset)
        for features in ((1.0,), (math.nan, 1.0), (math.inf, 1.0)):
            with self.subTest(features=features), self.assertRaises(ValueError):
                transform_features(features, scaler)
        with self.assertRaises(ValueError):
            transform_features((1.0, 2.0), replace(scaler, scales=(0.0, 1.0)))
        with self.assertRaises(ValueError):
            standardize_dataset(
                replace(self.dataset, feature_names=("constant", "varying")), scaler
            )


if __name__ == "__main__":
    unittest.main()

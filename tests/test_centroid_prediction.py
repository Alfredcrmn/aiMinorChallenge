import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nearest_centroid_from_scratch import (
    Centroid, Dataset, NearestCentroidModel, Sample, Standardizer,
    fit_nearest_centroid, predict, squared_euclidean_distance,
)


class CentroidPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Raw 12 becomes standardized 1. A second transformation would be wrong.
        self.model = NearestCentroidModel(
            Standardizer(("x",), (10.0,), (2.0,)),
            (Centroid(0, "left", 2, (-1.0,)), Centroid(1, "right", 2, (1.0,))),
        )

    def test_squared_distance_known_values(self) -> None:
        self.assertEqual(squared_euclidean_distance((0.0, 0.0), (3.0, 4.0)), 25.0)

    def test_predict_standardizes_raw_features_once(self) -> None:
        result = predict(self.model, (12.0,))
        self.assertEqual(result.class_id, 1)
        self.assertEqual(result.class_name, "right")
        self.assertEqual(result.squared_distance, 0.0)
        self.assertEqual(predict(self.model, (7.0,)).class_id, 0)

    def test_exact_tie_uses_smallest_class_id_regardless_of_order(self) -> None:
        model = replace(self.model, centroids=self.model.centroids[::-1])
        self.assertEqual(predict(model, (10.0,)).class_id, 0)

    def test_whole_pipeline_ignores_test_data_and_preserves_inputs(self) -> None:
        dataset = Dataset(
            ("x", "constant"), ("zero", "one", "two"),
            (Sample((0.0, 5.0), 0, 0, 0), Sample((2.0, 5.0), 0, 1, 0),
             Sample((10.0, 5.0), 1, 2, 0), Sample((12.0, 5.0), 1, 3, 0),
             Sample((20.0, 5.0), 2, 4, 0), Sample((22.0, 5.0), 2, 5, 0)),
            (Sample((1e10, -1e10), 2, 6, 0),),
        )
        model = fit_nearest_centroid(dataset)
        self.assertEqual(model, fit_nearest_centroid(replace(dataset, test=())))
        for class_id, value in enumerate((1.0, 11.0, 21.0)):
            self.assertEqual(predict(model, (value, 5.0)).class_id, class_id)
        self.assertEqual(dataset.train[0].features, (0.0, 5.0))

    def test_invalid_prediction_inputs(self) -> None:
        for features in ((), (1.0, 2.0), (math.nan,), (math.inf,)):
            with self.subTest(features=features), self.assertRaises(ValueError):
                predict(self.model, features)
        with self.assertRaises(ValueError):
            predict(replace(self.model, centroids=()), (1.0,))

    def test_invalid_distance_inputs(self) -> None:
        for left, right in (
            ((), ()), ((1.0,), (1.0, 2.0)), ((math.nan,), (0.0,)),
            ((1e308,), (-1e308,)),
        ):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                squared_euclidean_distance(left, right)


if __name__ == "__main__":
    unittest.main()

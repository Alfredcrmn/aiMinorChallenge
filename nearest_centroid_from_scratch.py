from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


METADATA_COLUMNS = (
    "record_id",
    "source_sample_id",
    "movement_id",
    "movement_type",
    "window_id",
    "start_timepoint",
    "end_timepoint",
    "split",
)
EXPECTED_FEATURE_COUNT = 72
EXPECTED_CLASS_COUNT = 16


@dataclass(frozen=True, slots=True)
class Sample:
    features: tuple[float, ...]
    label: int
    record_id: int
    window_id: int


@dataclass(frozen=True, slots=True)
class Dataset:
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    train: tuple[Sample, ...]
    test: tuple[Sample, ...]


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Training-only means and divisors, in the same order as feature_names."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]  # Population standard deviations; 1 for constants.


@dataclass(frozen=True, slots=True)
class Centroid:
    """Mean standardized feature vector for one movement class."""

    class_id: int
    class_name: str
    sample_count: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NearestCentroidModel:
    """Keep centroids with the training scaler needed for future inputs."""

    standardizer: Standardizer
    centroids: tuple[Centroid, ...]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Closest class and its squared Euclidean distance (not a probability)."""

    class_id: int
    class_name: str
    squared_distance: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Global accuracy and equally weighted class averages, as fractions."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def parse_integer(row: dict[str, str], column: str, line_number: int) -> int:
    """Read an integer column and report the exact location of invalid data."""
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid integer in column {column!r} at CSV line {line_number}"
        ) from error


def parse_features(
    row: dict[str, str], feature_names: tuple[str, ...], line_number: int
) -> tuple[float, ...]:
    """Convert the feature columns in one CSV row to finite floats."""
    values: list[float] = []

    for feature_name in feature_names:
        try:
            value = float(row[feature_name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid value for feature {feature_name!r} "
                f"at CSV line {line_number}"
            ) from error

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite value for feature {feature_name!r} "
                f"at CSV line {line_number}"
            )
        values.append(value)

    return tuple(values)


def validate_header(fieldnames: list[str] | None) -> tuple[str, ...]:
    """Validate the metadata prefix and return the ordered feature names."""
    if fieldnames is None:
        raise ValueError("The CSV file is empty or does not contain a header")
    if tuple(fieldnames[: len(METADATA_COLUMNS)]) != METADATA_COLUMNS:
        raise ValueError(
            "Unexpected CSV metadata columns. Expected the file to start with: "
            + ", ".join(METADATA_COLUMNS)
        )

    feature_names = tuple(fieldnames[len(METADATA_COLUMNS) :])
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns; "
            f"received {len(feature_names)}"
        )
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("The CSV header contains duplicate column names")

    return feature_names


def load_dataset(csv_path: Path) -> Dataset:
    """Load the feature CSV and preserve its recording-level train/test split."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    train_samples: list[Sample] = []
    test_samples: list[Sample] = []
    class_names: dict[int, str] = {}
    record_splits: dict[int, str] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        feature_names = validate_header(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            record_id = parse_integer(row, "record_id", line_number)
            window_id = parse_integer(row, "window_id", line_number)
            label = parse_integer(row, "movement_id", line_number)
            movement_type = row["movement_type"]
            split = row["split"]

            if not 0 <= label < EXPECTED_CLASS_COUNT:
                raise ValueError(
                    f"movement_id outside 0-{EXPECTED_CLASS_COUNT - 1} "
                    f"at CSV line {line_number}"
                )
            if not movement_type:
                raise ValueError(f"Empty movement_type at CSV line {line_number}")
            previous_name = class_names.setdefault(label, movement_type)
            if previous_name != movement_type:
                raise ValueError(
                    f"Inconsistent movement_type for class {label} "
                    f"at CSV line {line_number}"
                )
            if split not in {"train", "test"}:
                raise ValueError(
                    f"Invalid split {split!r} at CSV line {line_number}"
                )

            previous_split = record_splits.setdefault(record_id, split)
            if previous_split != split:
                raise ValueError(
                    f"record_id={record_id} appears in both train and test"
                )

            sample = Sample(
                features=parse_features(row, feature_names, line_number),
                label=label,
                record_id=record_id,
                window_id=window_id,
            )
            if split == "train":
                train_samples.append(sample)
            else:
                test_samples.append(sample)

    if not train_samples:
        raise ValueError("The dataset does not contain training samples")
    if not test_samples:
        raise ValueError("The dataset does not contain test samples")
    if set(class_names) != set(range(EXPECTED_CLASS_COUNT)):
        raise ValueError(
            f"Expected classes 0-{EXPECTED_CLASS_COUNT - 1}; "
            f"received {sorted(class_names)}"
        )

    ordered_class_names = tuple(
        class_names[class_id] for class_id in range(EXPECTED_CLASS_COUNT)
    )
    return Dataset(
        feature_names=feature_names,
        class_names=ordered_class_names,
        train=tuple(train_samples),
        test=tuple(test_samples),
    )


def fit_standardizer(dataset: Dataset) -> Standardizer:
    """Compute global statistics from train only, without using class labels."""
    if not dataset.train:
        raise ValueError("Cannot standardize an empty training set")
    feature_count = len(dataset.feature_names)
    if not feature_count:
        raise ValueError("Cannot standardize without features")

    for sample in dataset.train:
        if len(sample.features) != feature_count:
            raise ValueError("Training sample has an incorrect feature count")
        if any(not math.isfinite(value) for value in sample.features):
            raise ValueError("Training features must all be finite numbers")

    sample_count = len(dataset.train)
    means: list[float] = []
    scales: list[float] = []
    for feature_index in range(feature_count):
        mean = math.fsum(
            sample.features[feature_index] for sample in dataset.train
        ) / sample_count
        variance = math.fsum(
            (sample.features[feature_index] - mean) ** 2
            for sample in dataset.train
        ) / sample_count
        standard_deviation = math.sqrt(variance)
        means.append(mean)
        scales.append(standard_deviation if standard_deviation > 0.0 else 1.0)

    return Standardizer(
        feature_names=dataset.feature_names,
        means=tuple(means),
        scales=tuple(scales),
    )


def transform_features(
    features: tuple[float, ...], standardizer: Standardizer
) -> tuple[float, ...]:
    """Apply training statistics to one window: z = (x - mean) / scale."""
    count = len(standardizer.feature_names)
    if (
        len(features) != count
        or len(standardizer.means) != count
        or len(standardizer.scales) != count
    ):
        raise ValueError("Feature count does not match the standardizer")

    transformed: list[float] = []
    for value, mean, scale in zip(
        features, standardizer.means, standardizer.scales
    ):
        if (
            not math.isfinite(value)
            or not math.isfinite(mean)
            or not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError("Features and parameters must be finite; scales positive")
        standardized_value = (value - mean) / scale
        if not math.isfinite(standardized_value):
            raise ValueError("Standardization produced a non-finite value")
        transformed.append(standardized_value)
    return tuple(transformed)


def standardize_dataset(dataset: Dataset, standardizer: Standardizer) -> Dataset:
    """Transform both splits without refitting or modifying the original data."""
    if dataset.feature_names != standardizer.feature_names:
        raise ValueError("Feature names/order do not match the standardizer")

    def transform_samples(samples: tuple[Sample, ...]) -> tuple[Sample, ...]:
        return tuple(
            Sample(
                features=transform_features(sample.features, standardizer),
                label=sample.label,
                record_id=sample.record_id,
                window_id=sample.window_id,
            )
            for sample in samples
        )

    return Dataset(
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        train=transform_samples(dataset.train),
        test=transform_samples(dataset.test),
    )


def calculate_centroids(standardized_dataset: Dataset) -> tuple[Centroid, ...]:
    """Average standardized training vectors by class, never using test rows.

    The caller must standardize the dataset first using training-only statistics.
    Each coordinate is the sum of that feature within the class divided by the
    number of training windows in that class.
    """
    if not standardized_dataset.train:
        raise ValueError("Cannot calculate centroids without training samples")
    feature_count = len(standardized_dataset.feature_names)
    class_count = len(standardized_dataset.class_names)
    if not feature_count or not class_count:
        raise ValueError("Centroids require feature names and class names")

    grouped_samples: list[list[Sample]] = [[] for _ in range(class_count)]
    for sample in standardized_dataset.train:
        if not 0 <= sample.label < class_count:
            raise ValueError(f"Invalid training class: {sample.label}")
        if len(sample.features) != feature_count:
            raise ValueError("Training sample has an incorrect feature count")
        if any(not math.isfinite(value) for value in sample.features):
            raise ValueError("Training features must all be finite numbers")
        grouped_samples[sample.label].append(sample)

    centroids: list[Centroid] = []
    for class_id, class_name in enumerate(standardized_dataset.class_names):
        samples = grouped_samples[class_id]
        if not samples:
            raise ValueError(f"Class {class_id} has no training samples")
        features = tuple(
            math.fsum(sample.features[j] for sample in samples) / len(samples)
            for j in range(feature_count)
        )
        centroids.append(
            Centroid(
                class_id=class_id,
                class_name=class_name,
                sample_count=len(samples),
                features=features,
            )
        )
    return tuple(centroids)


def fit_nearest_centroid(dataset: Dataset) -> NearestCentroidModel:
    """Fit the scaler and centroids using raw training features only."""
    standardizer = fit_standardizer(dataset)
    training_only = Dataset(
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        train=dataset.train,
        test=(),
    )
    standardized_train = standardize_dataset(training_only, standardizer)
    return NearestCentroidModel(
        standardizer=standardizer,
        centroids=calculate_centroids(standardized_train),
    )


def squared_euclidean_distance(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    """Sum squared coordinate differences; no square root is needed to rank."""
    if not left or len(left) != len(right):
        raise ValueError("Distance requires non-empty vectors of equal length")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Distance requires finite coordinates")
    try:
        distance = math.fsum((x - y) ** 2 for x, y in zip(left, right))
    except OverflowError as error:
        raise ValueError("Coordinates are too large to calculate distance") from error
    if not math.isfinite(distance):
        raise ValueError("Distance is not finite")
    return distance


def predict(model: NearestCentroidModel, features: tuple[float, ...]) -> Prediction:
    """Standardize a raw feature vector once, then select the closest class.

    Input must use the same feature order as the training CSV. Do not pass an
    already standardized vector. Exact ties use the smallest class_id.
    """
    if not model.centroids:
        raise ValueError("Cannot predict without centroids")
    standardized = transform_features(features, model.standardizer)
    candidates = (
        Prediction(
            class_id=centroid.class_id,
            class_name=centroid.class_name,
            squared_distance=squared_euclidean_distance(
                standardized, centroid.features
            ),
        )
        for centroid in model.centroids
    )
    return min(candidates, key=lambda item: (item.squared_distance, item.class_id))


def calculate_accuracy(
    actual_labels: tuple[int, ...], predicted_labels: tuple[int, ...]
) -> float:
    """Return the fraction of matching labels, between zero and one."""
    if not actual_labels:
        raise ValueError("Accuracy requires at least one sample")
    if len(actual_labels) != len(predicted_labels):
        raise ValueError("Actual and predicted labels must have equal lengths")
    correct = sum(
        actual == predicted
        for actual, predicted in zip(actual_labels, predicted_labels)
    )
    return correct / len(actual_labels)


def calculate_classification_metrics(
    actual_labels: tuple[int, ...],
    predicted_labels: tuple[int, ...],
    class_count: int,
) -> ClassificationMetrics:
    """Calculate multiclass metrics manually over classes 0..class_count-1.

    Undefined class metrics (zero denominator) are assigned zero. All configured
    classes contribute to macro averages, even if absent from this sample set.
    Macro F1 is the average of class F1 scores, not the F1 of macro P and R.
    """
    if class_count <= 0:
        raise ValueError("Metrics require a positive class count")
    accuracy = calculate_accuracy(actual_labels, predicted_labels)
    true_positives = [0] * class_count
    actual_counts = [0] * class_count
    predicted_counts = [0] * class_count

    for actual, predicted in zip(actual_labels, predicted_labels):
        if not 0 <= actual < class_count or not 0 <= predicted < class_count:
            raise ValueError("Labels are outside the configured class range")
        actual_counts[actual] += 1
        predicted_counts[predicted] += 1
        if actual == predicted:
            true_positives[actual] += 1

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_id in range(class_count):
        tp = true_positives[class_id]
        # predicted_count = TP + FP; actual_count = TP + FN.
        predicted_count = predicted_counts[class_id]
        actual_count = actual_counts[class_id]
        precisions.append(tp / predicted_count if predicted_count else 0.0)
        recalls.append(tp / actual_count if actual_count else 0.0)
        # F1 = 2*TP / (2*TP + FP + FN), including the zero-TP case.
        denominator = actual_count + predicted_count
        f1_scores.append(2 * tp / denominator if denominator else 0.0)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=math.fsum(precisions) / class_count,
        recall_macro=math.fsum(recalls) / class_count,
        f1_macro=math.fsum(f1_scores) / class_count,
    )


def nonnegative_integer(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a non-negative integer") from error
    if count < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer")
    return count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_csv = (
        project_root
        / "REHAB"
        / "Rehab_exercise"
        / "d03_feature_data"
        / "rehab_exercise_features.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="Path to the derived feature CSV",
    )
    parser.add_argument(
        "--examples", type=nonnegative_integer, default=10,
        help="Number of test examples to display; metrics always use all test rows",
    )
    parser.add_argument(
        "--show-centroids", action="store_true",
        help="Also display training counts and first three centroid coordinates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        dataset = load_dataset(args.csv.resolve())
        model = fit_nearest_centroid(dataset)
        predictions = tuple(predict(model, sample.features) for sample in dataset.test)
        actual_labels = tuple(sample.label for sample in dataset.test)
        predicted_labels = tuple(prediction.class_id for prediction in predictions)
        metrics = calculate_classification_metrics(
            actual_labels, predicted_labels, len(dataset.class_names)
        )
        correct = sum(a == p for a, p in zip(actual_labels, predicted_labels))
        examples = list(zip(dataset.test[:args.examples], predictions[:args.examples]))
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None

    print(f"Dataset: {args.csv.resolve()}")
    print(f"Features per sample: {len(dataset.feature_names)}")
    print(f"Classes: {len(dataset.class_names)}")
    print(f"Training samples: {len(dataset.train):,}")
    print(f"Test samples: {len(dataset.test):,}")
    print("Scaler and centroids fitted using training samples only")
    print(f"Centroids calculated: {len(model.centroids)}")
    print(f"Accuracy: {metrics.accuracy * 100:.2f}% (test windows)")
    print(f"Precision (macro): {metrics.precision_macro * 100:.2f}%")
    print(f"Recall (macro): {metrics.recall_macro * 100:.2f}%")
    print(f"F1 score (macro): {metrics.f1_macro * 100:.2f}%")
    print(f"Correct predictions: {correct:,} / {len(dataset.test):,}")
    if args.show_centroids:
        print("Preview coordinates: " + ", ".join(dataset.feature_names[:3]))
        for centroid in model.centroids:
            preview = ", ".join(f"{value:.6f}" for value in centroid.features[:3])
            print(
                f"Class {centroid.class_id:02d} ({centroid.class_name}): "
                f"samples={centroid.sample_count:,}, "
                f"dimensions={len(centroid.features)}, first coordinates=({preview})"
            )
    print(f"Test predictions shown: {len(examples)}")
    for sample, prediction in examples:
        print(
            f"record_id={sample.record_id}, window_id={sample.window_id}, "
            f"actual={sample.label:02d}, predicted={prediction.class_id:02d} "
            f"({prediction.class_name}), squared_distance={prediction.squared_distance:.6f}"
        )


if __name__ == "__main__":
    main()

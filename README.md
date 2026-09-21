# aiMinorChallenge

Classify REHAB exercises from windowed sensor features.

## 1. Setup

Run these commands from the project folder. Python 3.12 was used for the recorded results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn
```

The three scripts only need these packages. The recorded environment used NumPy
2.5.3, pandas 2.3.3, scikit-learn 1.9.1, and SciPy 1.18.1; results may vary with
library versions.

Place the original processed `.npy` files in:

```text
REHAB/Rehab_exercise/d02_processed_data/
```

The scripts use paired files `000_1.npy` / `000_2.npy` through `015_1.npy` /
`015_2.npy`. **Both `014_1.npy` and `014_2.npy` are skipped.**

## 2. Build the dataset

```bash
python scripts/build_exercise_dataset.py
```

Creates `data/rehab_exercise.csv`: **34,056 rows**, one per window.
Each original recording is split into eight non-overlapping windows of 110 points.
Matching sample indices in the two sensor files are treated as the same recording.

- **72 feature columns:** six statistics for each of 12 channels: mean, standard
  deviation, RMS, range, mean absolute difference, and slope.
- **`exercise`:** target label (`0`–`13` and `15`).
- **`sample_id`:** exercise and original sample index.
- **`repetition_id`:** unique recording ID, shared by its eight windows.
- **`window`:** window number (`1`–`8`).

Here, a repetition ID identifies a source recording, not an individually detected
motion cycle. The supplied processed signals are used without additional scaling
or filtering during dataset creation.

## 3. Compare five models

```bash
python scripts/evaluate_models.py
```

Evaluates logistic regression, linear SVM, KNN, Random Forest, and Extra Trees
using **five-fold GroupKFold grouped by `repetition_id`**. All windows from a
recording stay in the same fold. IDs and window numbers are excluded from model
inputs; scaling for logistic regression, SVM, and KNN is fitted within each fold.

Creates `data/model_results.csv` with the fold mean and standard deviation of
accuracy, balanced accuracy, macro precision, macro recall, macro F1, and weighted F1.

## 4. Tune the best two models

```bash
python scripts/tune_models.py
```

Tunes Extra Trees and Random Forest using the same grouped folds, selecting the
highest mean macro F1. The search checks eight combinations per model:

```python
{
    "n_estimators": [100, 200],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 2],
}
```

Creates `data/tuning_results.csv` with the best parameters and the same metrics.
Both models use `random_state=42`. The recorded best parameters for both were:

```python
{"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1}
```

| Tuned model | Accuracy | Macro F1 |
|---|---:|---:|
| Extra Trees | 96.10% | 95.97% |
| Random Forest | 95.35% | 95.20% |

These are tuning cross-validation scores, not independent test scores. Grouping
prevents recordings from crossing folds, but cannot ensure patient separation
because patient IDs are unavailable. No trained model is saved.

Rerunning each script overwrites its corresponding output CSV.

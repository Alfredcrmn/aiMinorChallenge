# aiMinorChallenge

Classify REHAB exercises from windowed sensor features.

## Replicate the complete workflow

Run from the project folder (tested with Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_exercise_dataset.py
python scripts/create_eda.py
python scripts/validate_models.py
```

`requirements.txt` includes **fg-data-profiling**, which is required for the EDA
step as well as NumPy, pandas, and scikit-learn for the dataset/model scripts.
Use the activated environment for every command; the profiling import is
`from data_profiling import ProfileReport`.

Place the original processed `.npy` files under
`REHAB/Rehab_exercise/d02_processed_data/` before running. The files are not
downloaded by the scripts. Profiling and model searches can take several minutes.

### Recorded environments

| Run | Python and numerical/model libraries | Profiling |
|---|---|---|
| Historical all-data CV (before cleaning) | Python 3.12.14; NumPy 2.5.3; pandas 2.3.3; SciPy 1.18.1; scikit-learn 1.9.1 | Not used by those model runs |
| Current cleaned holdout workflow and EDA | Python 3.12.14; NumPy 2.5.3; pandas 2.3.3; SciPy 1.18.1; scikit-learn 1.9.1 | fg-data-profiling 4.20.0 |

These local runs used the same numerical/model library versions, but different
datasets and evaluation protocols. There is no separately locked historical
environment. Dependencies in `requirements.txt` are not pinned, so new
installations may produce slightly different results.

## Data preparation and cleaning

The original request specifically selected the **processed** data and excluded
**both `014_1.npy` and `014_2.npy`**. This is a scope exclusion, not a claim that
exercise 014 is clinically invalid or that both files failed quality checks.

The [source paper](https://doi.org/10.1038/s41597-026-07802-2), p. 9, describes
upstream cleaning: truncation or padding to 880 points, a moving-average filter
with a 10-point window, zero-mean normalization, and removal of erroneous samples.
These steps belong to the dataset authors. This repository does not reconstruct
raw acquisition cleaning or claim to have performed it. Filtering already
processed signals again would change the requested input.

Our dataset builder performs the following ETL:

1. **Extract:** load paired sensor arrays with `allow_pickle=False`, check matching
   sample counts and shape `(recordings, 880, 6)`.
2. **Clean:** remove an entire paired recording if either sensor contains NaN or
   infinity; deduplicate exact full paired signals within each exercise; then remove
   recordings that are entirely zero across both sensors. Preserve original sample
   indices. `data/cleaning_summary.csv` records **0 nonfinite removals, 627 duplicate
   removals, and 10 all-zero removals**, leaving **3,620 recordings**. These counts
   are sequential and do not overlap. All-zero recordings contain no usable signal;
   their underlying cause (padding, rest, or failed acquisition) is not inferred.
3. **Transform:** combine the two sensor configurations into 12 channels, split
   each recording into eight non-overlapping windows of 110 points (about 2.2 s
   at 50 Hz), and extract six statistics per channel.
4. **Load:** write `data/rehab_exercise.csv` with **28,960 rows and 76 columns**.

Each row contains 72 features (mean, population standard deviation, RMS, range,
mean absolute difference, and slope) plus:

| Column | Meaning |
|---|---|
| `exercise` | Target label: 0–13 and 15 |
| `sample_id` | Exercise and original sample index |
| `repetition_id` | Unique source recording ID, shared by all eight windows |
| `window` | Window number, 1–8 |

Matching sample indices across `_1` and `_2` are assumed to refer to one recording.
A recording ID is not a detected motion-cycle ID or a patient ID. Existing rest
and padding are preserved; no valid-length masks are supplied. Individual duplicate feature
vectors, extreme values, and zero windows within otherwise informative recordings
are not automatically removed; they may represent repeated or stationary movement.
Only exact whole-recording copies and entirely all-zero recordings are removed.
The final development/test split has no identical feature-window vectors in common.

## EDA report

Open [EDA/rehab_profile.html](EDA/rehab_profile.html) for the **fg-data-profiling**
report and [EDA/findings.md](EDA/findings.md) for a short interpretation.
The library is imported as `from data_profiling import ProfileReport`.

The report covers every development window, all 72 features, and the categorical
exercise label. It includes distributions, missingness, correlations, duplicates,
and automated alerts. Identifiers and window numbers are omitted. Test values
are excluded from EDA; pairwise scatter plots are disabled to keep the report
manageable. EDA does not automatically drop features or observations.

## Training, validation, and test evaluation

`validate_models.py` uses the reproducible split in `data_split.py`:

- Reserve **20% of recording IDs** for the final test, stratified by exercise,
  using `random_state=42`. The remaining 80% form the development set.
- Compare logistic regression, linear SVM, KNN, Random Forest, and Extra Trees
  using **five-fold GroupKFold within development only**. All windows from a
  recording stay together. Scaling is fitted inside the training folds.
- Select the top two models by development macro F1, then tune them on those
  development folds. The tree grid tests 100/200 trees, `max_features` of
  `sqrt`/0.5, and `min_samples_leaf` of 1/2.
- Choose the winner by validation macro F1, refit on all development recordings,
  then evaluate that single winner on the held-out test once.

IDs, window number, and the target are excluded from predictive features. Split
assertions check recording separation. Macro F1 is the primary metric because
it weights exercise classes equally. Reports also include accuracy, balanced
accuracy, macro precision/recall, and weighted F1.

Read [data/holdout/report.md](data/holdout/report.md) for the train/validation/test
comparison, best parameters, weakest classes, frequent confusions, and the
observed effect of regularization. Supporting CSVs in that folder contain split
assignments, development model comparisons, all tuning candidates, metrics,
class-level scores, and the confusion matrix (rows = actual, columns = predicted).

### Current recorded results

The cleaned dataset contains **3,620 recordings**: **2,896 development** and
**724 test** (23,168 and 5,792 windows respectively; **28,960 windows total**).
The selected model is Extra Trees with 200 trees, `max_features="sqrt"`, and
`min_samples_leaf=1`.

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Development training (fitted data) | 100.00% | 100.00% |
| Development validation (five-fold CV) | 95.67% | 95.43% |
| Held-out test | **95.32%** | **94.90%** |

Values come from [data/holdout/metrics.csv](data/holdout/metrics.csv). Training
scores are resubstitution scores, not generalization estimates.

**Evaluation limits:** this is a retrospective holdout. Earlier experiments used
all recordings, so it is not a pristine external cohort. The new run separates
test data from fitting, selection, tuning, and EDA, but cannot undo prior
exploration. Do not adjust models in response to this test score. Patient IDs
are unavailable, so this is recording-separated, not patient-independent,
evaluation. Window-level scores do not measure clinical benefit.

## Historical exploratory experiments

The original scripts remain available:

```bash
python scripts/evaluate_models.py
python scripts/tune_models.py
```

They use all-data grouped CV and produce `data/model_results.csv` and
`data/tuning_results.csv`. The existing files are **historical exploratory
estimates from the pre-cleaning dataset**, not current held-out results.
Rerunning these scripts now uses the current cleaned CSV and overwrites those
files; it does not reproduce the old dataset or old scores. Use
`validate_models.py` for the current development/validation/test workflow.
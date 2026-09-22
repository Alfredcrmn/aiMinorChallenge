# Evaluation results

Selected model: **Extra trees**. Parameters: `{"max_features": "sqrt", "min_samples_leaf": 1, "n_estimators": 200}`.

Development: 23,168 windows / 2,896 recordings. Test: 5,792 windows / 724 recordings.

The class-stratified 80/20 split is made on recording IDs before model selection. Five-fold GroupKFold within development supplies training and validation folds. The two leading families are tuned on development only; the winning model is refitted on all development recordings and evaluated once on the test set.

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| training (resubstitution) | 100.00% | 100.00% |
| validation (5-fold CV) | 95.67% | 95.43% |
| test (held-out recordings) | 95.32% | 94.90% |

Test balanced accuracy: 94.74%; macro precision: 95.15%; macro recall: 94.74%; weighted F1: 95.31%.

## Interpretation

Tuning changed development macro F1 from 95.15% to 95.43% (+0.28 percentage points). This comparison uses the same development folds; it is a selection gain, not proof of a statistically significant improvement. Macro F1 weights every exercise equally; weighted F1 gives larger classes more influence.

Training macro F1 exceeds test macro F1 by 5.10 percentage points. Validation exceeds test by 0.53 points. Perfect training performance therefore does not imply perfect generalization; the smaller validation/test gap is consistent with the development estimate on this split, but does not establish robustness across patients or alternative splits.

Lowest test F1 classes:
- Exercise 001: F1 88.34%, recall 88.49%, 304 windows.
- Exercise 002: F1 88.55%, recall 85.29%, 272 windows.
- Exercise 012: F1 91.56%, recall 88.12%, 320 windows.

Most frequent directional errors (actual → predicted):
- 002 → 001: 26 windows.
- 012 → 006: 19 windows.
- 008 → 009: 19 windows.
- 001 → 010: 16 windows.
- 003 → 010: 15 windows.

These errors identify exercise pairs for signal-level review; the confusion matrix alone cannot establish a clinical or biomechanical cause.

## Regularization

Larger minimum leaf sizes constrain tree complexity. Restricting max_features reduces the features available at each split and diversifies trees. More trees average more predictions but do not directly limit individual tree depth. The complete development search is in tuning_search.csv.
- Extra trees: mean CV macro F1 across the balanced grid was 95.27% for leaf size 1 versus 94.62% for leaf size 2.
- Random forest: mean CV macro F1 across the balanced grid was 93.65% for leaf size 1 versus 92.94% for leaf size 2.

## Limits

This is a retrospective holdout: earlier exploratory scripts used all recordings. The current run keeps test data out of fitting, selection, and tuning, but the holdout is not a pristine external validation cohort. Prior exploration cannot be undone. Do not tune further against this test score. Patient IDs are unavailable, so different recordings from one patient may span splits. Window scores are correlated within recordings and are not clinical outcome or patient-independent performance estimates.

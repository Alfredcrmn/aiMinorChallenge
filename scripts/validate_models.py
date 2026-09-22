"""Compare/tune on development recordings, then evaluate one winner on holdout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_split import split_recordings

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/holdout"


def main():
    data = pd.read_csv(ROOT / "data/rehab_exercise.csv")
    development, test = split_recordings(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assignments = data[["repetition_id", "exercise"]].drop_duplicates().copy()
    assignments["split"] = np.where(assignments.repetition_id.isin(
        data.loc[development, "repetition_id"]), "development", "test")
    assignments.to_csv(OUTPUT / "recording_splits.csv", index=False)
    features = data.drop(columns=["sample_id", "repetition_id", "window", "exercise"])
    X, y = features.loc[development], data.loc[development, "exercise"]
    groups = data.loc[development, "repetition_id"]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    for train, valid in folds:
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[valid])
        assert set(y.iloc[train]) == set(y) == set(y.iloc[valid])

    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(
            dual=False, max_iter=5000, random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Random forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    scoring = {"f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
               "accuracy": "accuracy"}
    comparisons = []
    for name, model in models.items():
        print(f"Development CV: {name}", flush=True)
        scores = cross_validate(model, X, y, cv=folds, scoring=scoring, error_score="raise")
        comparisons.append({"model": name, "f1_macro": scores["test_f1_macro"].mean(),
                            "accuracy": scores["test_accuracy"].mean()})
    comparison = pd.DataFrame(comparisons).sort_values("f1_macro", ascending=False)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)

    grids = {
        "Extra trees": {"n_estimators": [100, 200], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]},
        "Random forest": {"n_estimators": [100, 200], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]},
        "KNN": {"kneighborsclassifier__n_neighbors": [3, 5, 9], "kneighborsclassifier__weights": ["uniform", "distance"]},
        "Logistic regression": {"logisticregression__C": [0.1, 1, 10]},
        "Linear SVM": {"linearsvc__C": [0.1, 1, 10]},
    }
    searches, candidates = [], []
    # Choose the top two anew, using development data only.
    for name in comparison.head(2)["model"]:
        print(f"Tuning on development data: {name}", flush=True)
        search = GridSearchCV(models[name], grids[name], scoring=scoring, refit="f1_macro",
                              cv=folds, n_jobs=1, error_score="raise")
        search.fit(X, y)
        searches.append((name, search))
        for i, params in enumerate(search.cv_results_["params"]):
            candidates.append({"model": name, "params": json.dumps(params),
                "f1_macro": search.cv_results_["mean_test_f1_macro"][i],
                "accuracy": search.cv_results_["mean_test_accuracy"][i]})
        pd.DataFrame(candidates).to_csv(OUTPUT / "tuning_search.csv", index=False)
        print(f"  Best CV macro F1: {search.best_score_:.4f}; {search.best_params_}", flush=True)

    name, winner = max(searches, key=lambda pair: pair[1].best_score_)
    # Winner is fixed before accessing test features or predicting test labels.
    X_test, y_test = features.loc[test], data.loc[test, "exercise"]
    prediction = winner.best_estimator_.predict(X_test)
    labels = sorted(y.unique())
    report = classification_report(y_test, prediction, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame({str(c): report[str(c)] for c in labels}).T.rename_axis("exercise").to_csv(
        OUTPUT / "class_metrics.csv")
    matrix = confusion_matrix(y_test, prediction, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("actual_exercise").to_csv(
        OUTPUT / "confusion_matrix.csv")
    results = pd.DataFrame([
        {"split": "training (resubstitution)", "accuracy": accuracy_score(y, winner.best_estimator_.predict(X)),
         "f1_macro": f1_score(y, winner.best_estimator_.predict(X), average="macro")},
        {"split": "validation (5-fold CV)", "accuracy": winner.cv_results_["mean_test_accuracy"][winner.best_index_],
         "f1_macro": winner.best_score_},
        {"split": "test (held-out recordings)", "accuracy": accuracy_score(y_test, prediction),
         "f1_macro": report["macro avg"]["f1-score"]},
    ])
    results.to_csv(OUTPUT / "metrics.csv", index=False)

    baseline = float(comparison.set_index("model").loc[name, "f1_macro"])
    weakest = sorted(labels, key=lambda c: report[str(c)]["f1-score"])[:3]
    errors = sorted([(int(matrix[i, j]), labels[i], labels[j]) for i in range(len(labels))
                     for j in range(len(labels)) if i != j and matrix[i, j]], reverse=True)[:5]
    lines = ["# Evaluation results", "",
        f"Selected model: **{name}**. Parameters: `{json.dumps(winner.best_params_)}`.", "",
        f"Development: {int(development.sum()):,} windows / {groups.nunique():,} recordings. "
        f"Test: {int(test.sum()):,} windows / {data.loc[test, 'repetition_id'].nunique():,} recordings.", "",
        "The class-stratified 80/20 split is made on recording IDs before model selection. "
        "Five-fold GroupKFold within development supplies training and validation folds. "
        "The two leading families are tuned on development only; the winning model is refitted "
        "on all development recordings and evaluated once on the test set.", "",
        "| Split | Accuracy | Macro F1 |", "|---|---:|---:|"]
    for row in results.to_dict("records"):
        lines.append(f"| {row['split']} | {row['accuracy']:.2%} | {row['f1_macro']:.2%} |")
    lines += ["", f"Test balanced accuracy: {balanced_accuracy_score(y_test, prediction):.2%}; "
        f"macro precision: {report['macro avg']['precision']:.2%}; "
        f"macro recall: {report['macro avg']['recall']:.2%}; "
        f"weighted F1: {report['weighted avg']['f1-score']:.2%}.", "",
        "## Interpretation", "",
        f"Tuning changed development macro F1 from {baseline:.2%} to {winner.best_score_:.2%} "
        f"({100*(winner.best_score_-baseline):+.2f} percentage points). "
        "This comparison uses the same development folds; it is a selection gain, not proof "
        "of a statistically significant improvement. Macro F1 weights every exercise equally; "
        "weighted F1 gives larger classes more influence.", "",
        f"Training macro F1 exceeds test macro F1 by "
        f"{100*(results.iloc[0]['f1_macro']-results.iloc[2]['f1_macro']):.2f} percentage points. "
        f"Validation exceeds test by {100*(winner.best_score_-results.iloc[2]['f1_macro']):.2f} "
        "points. Perfect training performance therefore does not imply perfect generalization; "
        "the smaller validation/test gap is consistent with the development estimate on this split, "
        "but does not establish robustness across patients or alternative splits.", "",
        "Lowest test F1 classes:"]
    lines += [f"- Exercise {c:03d}: F1 {report[str(c)]['f1-score']:.2%}, "
              f"recall {report[str(c)]['recall']:.2%}, {int(report[str(c)]['support'])} windows."
              for c in weakest]
    lines += ["", "Most frequent directional errors (actual → predicted):"]
    lines += [f"- {a:03d} → {b:03d}: {n} windows." for n, a, b in errors]
    lines += ["", "These errors identify exercise pairs for signal-level review; the confusion "
              "matrix alone cannot establish a clinical or biomechanical cause.", "",
              "## Regularization", "",
              "Larger minimum leaf sizes constrain tree complexity. Restricting max_features "
              "reduces the features available at each split and diversifies trees. More trees "
              "average more predictions but do not directly limit individual tree depth. "
              "The complete development search is in tuning_search.csv."]
    for model_name, search in searches:
        records = pd.DataFrame(search.cv_results_["params"])
        if "min_samples_leaf" in records:
            records["f1"] = search.cv_results_["mean_test_f1_macro"]
            means = records.groupby("min_samples_leaf")["f1"].mean()
            lines.append(f"- {model_name}: mean CV macro F1 across the balanced grid was "
                         f"{means[1]:.2%} for leaf size 1 versus {means[2]:.2%} for leaf size 2.")
    lines += ["", "## Limits", "",
        "This is a retrospective holdout: earlier exploratory scripts used all recordings. "
        "The current run keeps test data out of fitting, selection, and tuning, but the holdout "
        "is not a pristine external validation cohort. Prior exploration cannot be undone. "
        "Do not tune further against this test score. Patient IDs are unavailable, so different "
        "recordings from one patient may span splits. Window scores are correlated within "
        "recordings and are not clinical outcome or patient-independent performance estimates."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")
    print(results.to_string(index=False), flush=True)
    print(f"Saved evaluation and interpretation to {OUTPUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

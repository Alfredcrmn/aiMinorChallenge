"""Compare five classifiers using five-fold GroupKFold by repetition_id."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = pd.read_csv(ROOT / "data/rehab_exercise.csv")
    X = data.drop(columns=["sample_id", "repetition_id", "window", "exercise"])
    y = data["exercise"]
    groups = data["repetition_id"]

    # Reuse the same folds for every model; a repetition never crosses folds.
    # These are recording groups, not patient groups (patient IDs are unavailable).
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    for train, test in folds:
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])

    # Pipelines fit scaling only on each training fold.
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000)),
        "Linear SVM": make_pipeline(
            StandardScaler(), LinearSVC(dual=False, max_iter=5000, random_state=42)),
        "KNN": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Random forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
    }
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    rows = []
    for name, model in models.items():
        print(f"Evaluating {name}...", flush=True)
        scores = cross_validate(model, X, y, cv=folds, scoring=scoring,
                                n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
        print(f"  F1 macro: {row['f1_macro_mean']:.4f}; "
              f"accuracy: {row['accuracy_mean']:.4f}", flush=True)

    results = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    output = ROOT / "data/model_results.csv"
    results.to_csv(output, index=False)
    print("\nFive-fold means:")
    print(results[["model"] + [f"{m}_mean" for m in scoring]].round(4).to_string(index=False))
    print(f"\nSaved means and standard deviations to {output}")


if __name__ == "__main__":
    main()

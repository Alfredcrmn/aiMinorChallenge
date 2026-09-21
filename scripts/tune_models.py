"""Tune Extra Trees and Random Forest with GroupKFold; optimize macro F1."""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = pd.read_csv(ROOT / "data/rehab_exercise.csv")
    X = data.drop(columns=["sample_id", "repetition_id", "window", "exercise"])
    y = data["exercise"]
    groups = data["repetition_id"]
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    for train, test in folds:
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])

    models = {
        "Extra trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "Random forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    }
    # Eight combinations per model, including the original baseline settings.
    grid = {
        "n_estimators": [100, 200],
        "max_features": ["sqrt", 0.5],
        "min_samples_leaf": [1, 2],
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
    output = ROOT / "data/tuning_results.csv"
    for name, model in models.items():
        print(f"Tuning {name}: 8 combinations x 5 folds...", flush=True)
        search = GridSearchCV(model, grid, scoring=scoring, refit=False,
                              cv=folds, n_jobs=1, error_score="raise")
        search.fit(X, y)
        results = search.cv_results_
        best = results["mean_test_f1_macro"].argmax()
        row = {"model": name, "best_params": json.dumps(results["params"][best])}
        for metric in scoring:
            values = pd.Series([results[f"split{i}_test_{metric}"][best] for i in range(5)])
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
        pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).to_csv(output, index=False)
        print(f"  Best parameters: {row['best_params']}", flush=True)
        print(f"  CV F1 macro: {row['f1_macro_mean']:.4f}; "
              f"accuracy: {row['accuracy_mean']:.4f}", flush=True)

    print(f"\nSaved best parameters and metrics to {output}")
    # These folds select parameters, so their scores are tuning estimates, not
    # an independent test. Groups represent recordings, not patient identities.
    print("Scores are tuning CV estimates; no independent test set was evaluated.")


if __name__ == "__main__":
    main()

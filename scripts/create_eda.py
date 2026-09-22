"""Generate an fg-data-profiling HTML report for development data only."""

from pathlib import Path

import pandas as pd
import numpy as np
from data_profiling import ProfileReport

from data_split import split_recordings

ROOT = Path(__file__).resolve().parents[1]


def write_findings(data):
    development, _ = split_recordings(data)
    frame = data.loc[development]
    features = frame.drop(columns=["sample_id", "repetition_id", "window", "exercise"])
    counts = frame.groupby("exercise").agg(recordings=("repetition_id", "nunique"), windows=("window", "size"))
    correlation = features.corr().abs()
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    lines = ["# EDA findings", "",
        "Generated with the same development/test split as validate_models.py. "
        "The HTML profile uses fg-data-profiling on every development window, "
        "with all 72 features and a categorical exercise label. IDs and window "
        "numbers are excluded from profiling. Test values are not profiled.", "",
        f"- Development: {len(frame):,} windows, {frame.repetition_id.nunique():,} recordings, {len(counts)} classes.",
        f"- Missing feature values: {int(features.isna().sum().sum())}; infinite values: {int(np.isinf(features).sum().sum())}.",
        f"- Constant feature columns: {int(features.nunique().le(1).sum())}.",
        f"- Duplicate feature vectors (beyond first occurrence, ignoring IDs and labels): {int(features.duplicated().sum())}.",
        f"- Largest/smallest class ratio: {counts.windows.max()/counts.windows.min():.2f}.", "",
        "Class counts are shown at both recording and window level; windows are "
        "correlated observations, not independent participants. Macro F1 prevents "
        "larger exercise classes from dominating the primary score.", "",
        "| Exercise | Recordings | Windows |", "|---|---:|---:|"]
    lines += [f"| {int(c):03d} | {int(row.recordings)} | {int(row.windows)} |" for c, row in counts.iterrows()]
    lines += ["", "## Feature dependence", "", "Largest absolute Pearson correlations:"]
    lines += [f"- `{a}` / `{b}`: {value:.4f}." for (a, b), value in pairs.head(5).items()]
    lines += ["", "Mean, standard deviation and RMS are mathematically related "
        "(RMS² = mean² + standard deviation²). Correlated descriptors can be "
        "redundant, so 72 columns do not imply 72 independent measurements. "
        "No feature selection was based on this report. Any future selection "
        "must be fitted within training folds.", "",
        "Zero or constant windows can reflect rest or source padding, while extreme "
        "values may reflect real movement. Without valid-length masks or clinical "
        "quality labels, they are not automatically deleted or clipped. Likewise, "
        "identical feature vectors do not establish duplicate recordings.", "",
        "The HTML report includes distributions, missing-value summaries, Pearson "
        "correlations, duplicate checks, and automated alerts. Pairwise scatter "
        "plots are disabled to keep the 72-feature report manageable."]
    output = ROOT / "EDA"
    output.mkdir(parents=True, exist_ok=True)
    (output / "findings.md").write_text("\n".join(lines) + "\n")


def main():
    data = pd.read_csv(ROOT / "data/rehab_exercise.csv")
    write_findings(data)
    development, _ = split_recordings(data)
    # Keep test values out of exploratory decisions. IDs are bookkeeping, not features.
    frame = data.loc[development].drop(columns=["sample_id", "repetition_id", "window"]).copy()
    frame["exercise"] = frame["exercise"].astype("category")
    report = ProfileReport(
        frame, title="REHAB exercise EDA — development recordings (80%)",
        explorative=True, pool_size=2,
        interactions={"continuous": False},
        correlations={"auto": {"calculate": False}, "pearson": {"calculate": True}},
        missing_diagrams={"bar": True, "matrix": False, "heatmap": False},
    )
    output = ROOT / "EDA/rehab_profile.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_file(output)
    print(f"Profiled all {len(frame):,} development windows; saved {output}")


if __name__ == "__main__":
    main()

"""Profile REHAB values without cleaning or transforming the source files.

Run from aiMinorChallenge:
    python profile_rehab_data.py
    python profile_rehab_data.py --output rehab_data_profile.md
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REHAB_DIR = SCRIPT_DIR / "REHAB"


def dataset_stage(path: Path) -> str:
    """Return a useful group name such as Rehab_assessment/d02_processed_data."""
    relative_parts = path.relative_to(REHAB_DIR).parts
    if path.name == "Label.npy":
        return "Rehab_assessment/Label.npy"
    if len(relative_parts) >= 2:
        return "/".join(relative_parts[:2])
    return relative_parts[0] if relative_parts else "unknown"


def axis_values(data: np.ndarray) -> np.ndarray:
    """Return rows x variables for 1D, 2D, or 3D arrays.

    For 3D sensor data, the variables are the final-axis channels and the
    first two axes are treated as observations over time for profiling only.
    The source array is not modified.
    """
    if data.ndim == 1:
        return data.reshape(-1, 1)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data.reshape(-1, data.shape[-1])
    raise ValueError(f"Unsupported array dimensions: {data.ndim}")


def update_stats(
    stats: dict[str, dict[str, np.ndarray | int]],
    key: str,
    data: np.ndarray,
) -> None:
    """Accumulate per-variable statistics for one dataset group."""
    values = axis_values(data)
    variables = values.shape[1]
    if key not in stats:
        stats[key] = {
            "files": 0,
            "variables": variables,
            "count": np.zeros(variables, dtype=np.int64),
            "non_finite": np.zeros(variables, dtype=np.int64),
            "sum": np.zeros(variables, dtype=np.float64),
            "sum_squares": np.zeros(variables, dtype=np.float64),
            "minimum": np.full(variables, np.inf),
            "maximum": np.full(variables, -np.inf),
        }

    group = stats[key]
    group["files"] = int(group["files"]) + 1
    finite = np.isfinite(values)
    group["count"] += finite.sum(axis=0)
    group["non_finite"] += (~finite).sum(axis=0)

    finite_values = np.where(finite, values, 0.0)
    group["sum"] += finite_values.sum(axis=0)
    group["sum_squares"] += np.square(finite_values).sum(axis=0)

    minimum = np.where(finite, values, np.inf).min(axis=0)
    maximum = np.where(finite, values, -np.inf).max(axis=0)
    group["minimum"] = np.minimum(group["minimum"], minimum)
    group["maximum"] = np.maximum(group["maximum"], maximum)


def format_number(value: float) -> str:
    """Format a statistic compactly for Markdown."""
    if not np.isfinite(value):
        return "-"
    return f"{value:.6g}"


def write_report(
    output_path: Path,
    file_count: int,
    readable_count: int,
    unreadable: list[tuple[str, str]],
    stage_counts: Counter[str],
    shape_counts: Counter[tuple[str, str]],
    dtype_counts: Counter[tuple[str, str]],
    stats: dict[str, dict[str, np.ndarray | int]],
    label_counts: list[dict[int, int]] | None,
) -> None:
    """Write the profiling results as a human-readable Markdown document."""
    with output_path.open("w", encoding="utf-8") as report:
        report.write("# REHAB Data Profile\n\n")
        report.write(
            "This report describes observed values. No source data was cleaned "
            "or transformed.\n\n"
        )
        report.write("## Scan status\n\n")
        report.write(f"- Files scanned: **{file_count}**\n")
        report.write(f"- Readable files: **{readable_count}**\n")
        report.write(f"- Unreadable files: **{len(unreadable)}**\n\n")

        report.write("## Files by dataset stage\n\n")
        report.write("| Dataset stage | Files |\n|---|---:|\n")
        for stage, count in sorted(stage_counts.items()):
            report.write(f"| `{stage}` | {count} |\n")
        report.write("\n")

        report.write("## Observed shapes\n\n")
        report.write("| Dataset stage | Shape | Files |\n|---|---|---:|\n")
        for (stage, shape), count in sorted(shape_counts.items()):
            report.write(f"| `{stage}` | `{shape}` | {count} |\n")
        report.write("\n")

        report.write("## Observed data types\n\n")
        report.write("| Dataset stage | Data type | Files |\n|---|---|---:|\n")
        for (stage, dtype), count in sorted(dtype_counts.items()):
            report.write(f"| `{stage}` | `{dtype}` | {count} |\n")
        report.write("\n")

        if label_counts is not None:
            report.write("## Label values by column\n\n")
            report.write("| Column | Observed value counts |\n|---:|---|\n")
            for column, counts in enumerate(label_counts):
                values = ", ".join(f"{value}: {count}" for value, count in sorted(counts.items()))
                report.write(f"| {column} | `{values}` |\n")
            report.write("\n")

        report.write("## Per-variable value statistics\n\n")
        report.write(
            "For 2D files, variables are columns. For 3D files, variables are "
            "the final-axis sensor channels.\n\n"
        )
        for key, group in sorted(stats.items()):
            count = group["count"]
            total = group["sum"]
            sum_squares = group["sum_squares"]
            mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
            variance = np.divide(sum_squares, count, out=np.full_like(total, np.nan), where=count > 0) - np.square(mean)
            standard_deviation = np.sqrt(np.maximum(variance, 0))

            report.write(f"### `{key}`\n\n")
            report.write(f"Files profiled: {group['files']}\n\n")
            report.write(
                "| Variable | Finite values | Non-finite values | Minimum | Maximum | Mean | Std. dev. |\n"
                "|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            for variable in range(int(group["variables"])):
                report.write(
                    f"| {variable} | {count[variable]} | {group['non_finite'][variable]} | "
                    f"{format_number(group['minimum'][variable])} | "
                    f"{format_number(group['maximum'][variable])} | "
                    f"{format_number(mean[variable])} | "
                    f"{format_number(standard_deviation[variable])} |\n"
                )
            report.write("\n")

        report.write("## Unreadable files\n\n")
        if unreadable:
            for path, error in unreadable:
                report.write(f"- `{path}`: {error}\n")
        else:
            report.write("None found.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "rehab_data_profile.md",
        help="Markdown report path (default: rehab_data_profile.md).",
    )
    args = parser.parse_args()

    files = sorted(
        path
        for path in REHAB_DIR.rglob("*.npy")
        if ".venv" not in path.parts
    )
    if not files:
        raise SystemExit(f"No .npy files found in {REHAB_DIR}")

    stage_counts: Counter[str] = Counter()
    shape_counts: Counter[tuple[str, str]] = Counter()
    dtype_counts: Counter[tuple[str, str]] = Counter()
    stats: dict[str, dict[str, np.ndarray | int]] = {}
    unreadable: list[tuple[str, str]] = []
    label_counts: list[dict[int, int]] | None = None
    readable_count = 0

    for path in files:
        relative_path = str(path.relative_to(REHAB_DIR))
        try:
            data = np.load(path, allow_pickle=False)
            if data.ndim not in (1, 2, 3) or not np.issubdtype(data.dtype, np.number):
                raise ValueError(f"unsupported array: ndim={data.ndim}, dtype={data.dtype}")
        except (OSError, ValueError, EOFError) as error:
            unreadable.append((relative_path, str(error)))
            continue

        readable_count += 1
        stage = dataset_stage(path)
        stage_counts[stage] += 1
        shape_counts[(stage, str(data.shape))] += 1
        dtype_counts[(stage, str(data.dtype))] += 1
        stats_key = f"{stage} | {data.ndim}D | {data.shape[-1] if data.ndim > 1 else 1} variables"
        update_stats(stats, stats_key, data)

        if path.name == "Label.npy" and data.ndim == 2:
            label_counts = []
            for column in range(data.shape[1]):
                values, counts = np.unique(data[:, column], return_counts=True)
                label_counts.append({int(value): int(count) for value, count in zip(values, counts)})

    output_path = args.output if args.output.is_absolute() else SCRIPT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        output_path,
        len(files),
        readable_count,
        unreadable,
        stage_counts,
        shape_counts,
        dtype_counts,
        stats,
        label_counts,
    )

    print(f"Files profiled: {len(files)}")
    print(f"Readable files: {readable_count}")
    print(f"Unreadable files: {len(unreadable)}")
    print(f"Profile written to: {output_path}")


if __name__ == "__main__":
    main()

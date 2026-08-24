"""Create a read-only inventory of the REHAB NumPy dataset.

Run from aiMinorChallenge:
    python inventory_rehab.py
    python inventory_rehab.py --output rehab_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REHAB_DIR = SCRIPT_DIR / "REHAB"


def dataset_group(path: Path) -> str:
    """Return the top-level REHAB folder containing a file."""
    relative_parts = path.relative_to(REHAB_DIR).parts
    return relative_parts[0] if relative_parts else "unknown"


def describe_file(path: Path) -> dict[str, str | int | float]:
    """Load one array and return descriptive metadata without changing it."""
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        return {
            "file": str(path.relative_to(REHAB_DIR)),
            "dataset_group": dataset_group(path),
            "status": "unreadable",
            "error": str(error),
            "shape": "",
            "dimensions": "",
            "dtype": "",
            "elements": "",
            "unique_values": "",
            "non_finite_values": "",
            "minimum": "",
            "maximum": "",
        }
    numeric = np.issubdtype(data.dtype, np.number)

    if numeric:
        finite = np.isfinite(data)
        non_finite_count = int((~finite).sum())
        finite_values = data[finite]
        minimum = float(finite_values.min()) if finite_values.size else ""
        maximum = float(finite_values.max()) if finite_values.size else ""
    else:
        non_finite_count = ""
        minimum = ""
        maximum = ""

    return {
        "file": str(path.relative_to(REHAB_DIR)),
        "dataset_group": dataset_group(path),
        "status": "readable",
        "error": "",
        "shape": str(data.shape),
        "dimensions": data.ndim,
        "dtype": str(data.dtype),
        "elements": data.size,
        "unique_values": np.unique(data).size,
        "non_finite_values": non_finite_count,
        "minimum": minimum,
        "maximum": maximum,
    }


def write_summary(records: list[dict[str, str | int | float]], output_path: Path) -> None:
    """Write a human-readable Markdown summary beside the detailed CSV."""
    readable = [record for record in records if record["status"] == "readable"]
    unreadable = [record for record in records if record["status"] == "unreadable"]

    groups = Counter(str(record["dataset_group"]) for record in records)
    shapes = Counter(
        (str(record["dataset_group"]), str(record["shape"]))
        for record in readable
    )
    dtypes = Counter(
        (str(record["dataset_group"]), str(record["dtype"]))
        for record in readable
    )

    with output_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("# REHAB Dataset Inventory Summary\n\n")
        summary_file.write(
            "This report describes the files without cleaning or transforming them.\n\n"
        )
        summary_file.write("## Overall status\n\n")
        summary_file.write(f"- Total files: **{len(records)}**\n")
        summary_file.write(f"- Readable files: **{len(readable)}**\n")
        summary_file.write(f"- Unreadable files: **{len(unreadable)}**\n\n")

        summary_file.write("## Files by dataset group\n\n")
        summary_file.write("| Dataset group | Files |\n|---|---:|\n")
        for group, count in sorted(groups.items()):
            summary_file.write(f"| `{group}` | {count} |\n")
        summary_file.write("\n")

        summary_file.write("## Shapes by dataset group\n\n")
        summary_file.write("| Dataset group | Shape | Files |\n|---|---|---:|\n")
        for (group, shape), count in sorted(shapes.items()):
            summary_file.write(f"| `{group}` | `{shape}` | {count} |\n")
        summary_file.write("\n")

        summary_file.write("## Data types by dataset group\n\n")
        summary_file.write("| Dataset group | Data type | Files |\n|---|---|---:|\n")
        for (group, dtype), count in sorted(dtypes.items()):
            summary_file.write(f"| `{group}` | `{dtype}` | {count} |\n")
        summary_file.write("\n")

        summary_file.write("## Unreadable files\n\n")
        if unreadable:
            for record in unreadable:
                summary_file.write(f"- `{record['file']}`: {record['error']}\n")
        else:
            summary_file.write("None found.\n")
        summary_file.write("\n")

        summary_file.write("## Representative readable files\n\n")
        summary_file.write("| Dataset group | File | Shape | Data type |\n|---|---|---|---|\n")
        seen_groups: set[str] = set()
        for record in readable:
            group = str(record["dataset_group"])
            if group not in seen_groups:
                summary_file.write(
                    f"| `{group}` | `{record['file']}` | `{record['shape']}` "
                    f"| `{record['dtype']}` |\n"
                )
                seen_groups.add(group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "rehab_inventory.csv",
        help="CSV report path (default: rehab_inventory.csv).",
    )
    args = parser.parse_args()

    files = sorted(
        path
        for path in REHAB_DIR.rglob("*.npy")
        if ".venv" not in path.parts
    )
    if not files:
        raise SystemExit(f"No .npy files found in {REHAB_DIR}")

    records = [describe_file(path) for path in files]
    fieldnames = list(records[0])
    output_path = args.output if args.output.is_absolute() else SCRIPT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.md")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    write_summary(records, summary_path)

    readable = [record for record in records if record["status"] == "readable"]
    unreadable = [record for record in records if record["status"] == "unreadable"]

    print(f"Files inventoried: {len(records)}")
    print(f"Readable files: {len(readable)}")
    print(f"Unreadable files: {len(unreadable)}")
    print(f"Inventory written to: {output_path}")
    print(f"Readable summary written to: {summary_path}")
    print("\nFiles by dataset group:")
    groups: dict[str, int] = {}
    for record in records:
        group = str(record["dataset_group"])
        groups[group] = groups.get(group, 0) + 1
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count}")

    print("\nFirst five files:")
    for record in records[:5]:
        print(
            f"  {record['file']} | status={record['status']} "
            f"| shape={record['shape']} | dtype={record['dtype']}"
        )

    if unreadable:
        print("\nUnreadable files (investigate before cleaning):")
        for record in unreadable:
            print(f"  {record['file']}: {record['error']}")


if __name__ == "__main__":
    main()

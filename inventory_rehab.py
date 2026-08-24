"""Create a read-only inventory of the REHAB NumPy dataset.

Run from aiMinorChallenge:
    python inventory_rehab.py
    python inventory_rehab.py --output rehab_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
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

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    readable = [record for record in records if record["status"] == "readable"]
    unreadable = [record for record in records if record["status"] == "unreadable"]

    print(f"Files inventoried: {len(records)}")
    print(f"Readable files: {len(readable)}")
    print(f"Unreadable files: {len(unreadable)}")
    print(f"Inventory written to: {output_path}")
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

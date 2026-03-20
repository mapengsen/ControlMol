from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_TASKS = ["ESOL", "BBBP", "hERG", "lipop", "Mutagenicity"]


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe stem."""
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "task"


def split_tasks(
    input_path: Path,
    output_dir: Path,
    tasks: Iterable[str],
    task_column: str = "task_name",
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> list[Path]:
    """Split the CSV into per-task files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if task_column not in df.columns:
        raise ValueError(f"Column {task_column!r} not found in {input_path.name}")

    if filter_column:
        if filter_column not in df.columns:
            raise ValueError(f"Column {filter_column!r} not found in {input_path.name}")
        filter_value_str = (filter_value or "").strip()
        df = df[df[filter_column].astype(str).str.strip() == filter_value_str]
        if df.empty:
            print(
                f"No rows match {filter_column} == {filter_value_str!r}; nothing to save."
            )
            return []

    output_dir.mkdir(parents=True, exist_ok=True)
    task_series = df[task_column].astype(str).str.strip()
    created: list[Path] = []

    for task in tasks:
        mask = task_series == task
        if not mask.any():
            print(f"Skip {task}: no matching rows.")
            continue

        subset = df[mask]
        output_path = output_dir / f"{input_path.stem}_{sanitize_filename(task)}.csv"
        subset.to_csv(output_path, index=False)
        created.append(output_path)
        print(f"Saved {task} -> {output_path} (rows: {len(subset)})")

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Prompt_MolOpt_train_val_ADMET.csv by task_name."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "Prompt_MolOpt_train_val_ADMET.csv",
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store per-task CSV files (defaults to the input file directory).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="task_name values to export (space separated).",
    )
    parser.add_argument(
        "--task-column",
        default="task_name",
        help="Column name containing task labels.",
    )
    parser.add_argument(
        "--filter-column",
        default="re_group",
        help="Column used for pre-filtering rows (e.g., re_group).",
    )
    parser.add_argument(
        "--filter-value",
        default="test_1",
        help="Value in filter-column to keep (e.g., test_1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent
    created = split_tasks(
        args.input,
        output_dir,
        args.tasks,
        args.task_column,
        args.filter_column,
        args.filter_value,
    )

    if not created:
        print("No CSV files were created.")
        return

    print("Created files:")
    for path in created:
        print(f"- {path}")


if __name__ == "__main__":
    main()

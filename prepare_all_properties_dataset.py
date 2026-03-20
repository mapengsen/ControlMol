from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DEFAULT_PROPERTIES = ["BBBP", "ESOL", "hERG", "lipop", "Mutagenicity"]
DEFAULT_PATTERN = "Prompt_MolOpt_train_val_ADMET_{prop}.csv"


def build_input_paths(input_dir: Path, properties: Iterable[str], pattern: str) -> List[Path]:
    paths: List[Path] = []
    for prop in properties:
        filename = pattern.format(prop=prop)
        paths.append(input_dir / filename)
    return paths


def validate_columns(df_columns: List[str], expected: List[str], path: Path) -> None:
    missing = [c for c in expected if c not in df_columns]
    extra = [c for c in df_columns if c not in expected]
    if missing or extra:
        raise ValueError(
            f"Columnsconsistent: {path} Missing {missing}，Make it more. {extra}"
        )


def merge_property_files(
    input_paths: List[Path],
    output_path: Path,
    label_column: str,
    fill_task_name: bool = True,
) -> None:
    if not input_paths:
        raise ValueError("No input file provided。")

    dfs = []
    expected_columns: List[str] | None = None

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        df = pd.read_csv(path)
        if expected_columns is None:
            expected_columns = list(df.columns)
        else:
            validate_columns(list(df.columns), expected_columns, path)

        prop = path.stem.split("_")[-1]
        df[label_column] = prop

        if fill_task_name and "task_name" in df.columns:
            df["task_name"] = df["task_name"].fillna(prop)

        dfs.append(df)
        print(f"Read {path} ({len(df)} CLI)，Characteristics of markings: {prop}")

    merged = pd.concat(dfs, ignore_index=True)

    ordered_columns = (
        [label_column] + expected_columns
        if label_column not in expected_columns
        else expected_columns
    )
    merged = merged.loc[:, ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(merged)} CLI {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge data/train_valid_data Five properties. CSV，Add target_property Columns，"
            "Generate all_properties_success_and_sampled_failures.csv。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/train_valid_data"),
        help="Single Properties CSV available。",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=DEFAULT_PROPERTIES,
        help="MergeIt is. .. Columns。",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Enter File Naming Mode（Use {prop} Placeholders）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/train_valid_data/all_properties_success_and_sampled_failures.csv"),
        help="MergeDocumentationPath。",
    )
    parser.add_argument(
        "--label-column",
        default="target_property",
        help="Characteristics of markingsIt is. .. Columns。",
    )
    parser.add_argument(
        "--no-fill-task-name",
        action="store_true",
        help="Do Not Autofill task_name。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = build_input_paths(Path(args.input_dir), args.properties, args.pattern)
    merge_property_files(
        input_paths=input_paths,
        output_path=Path(args.output),
        label_column=args.label_column,
        fill_task_name=not args.no_fill_task_name,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_column(df: pd.DataFrame, column: str, path: Path) -> pd.Series:
    """Return a trimmed column, raising if it is missing."""
    if column not in df.columns:
        raise ValueError(f"Missing column '{column}' in {path}")
    series = df[column].fillna("").astype(str).str.strip()
    return series[series != ""]


def merge_smiles(
    train_path: Path, test_path: Path, output_path: Path, dedup: bool = True
) -> int:
    """Merge train start/final and test start columns into one smiles CSV."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    smiles_parts = [
        _load_column(train_df, "start", train_path),
        _load_column(train_df, "final", train_path),
        _load_column(test_df, "start", test_path),
    ]

    smiles = pd.concat(smiles_parts, ignore_index=True)
    if dedup:
        smiles = smiles.drop_duplicates().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    smiles.to_csv(output_path, index=False, header=["smiles"])
    return len(smiles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine ADMET CSVs into a single smiles column."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/Prompt_MolOpt_train_val_ADMET.csv"),
        help="Training CSV with start/final columns.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/test_for_ADMET.csv"),
        help="Test CSV with start column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/merged_admet_smiles.csv"),
        help="Destination CSV path with a single smiles column.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicated smiles instead of dropping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count = merge_smiles(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        dedup=not args.keep_duplicates,
    )
    status = "deduplicated" if not args.keep_duplicates else "with duplicates"
    print(f"Saved {row_count} {status} smiles to {args.output}")


if __name__ == "__main__":
    main()

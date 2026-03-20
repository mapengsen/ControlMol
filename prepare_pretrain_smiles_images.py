from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SMILES CSV file into molecular PNG images for diffusion pretraining."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/pretrain_imageGeneration.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pretrain_imageGeneration_images/image_file",
        help="Directory used to save generated PNG images.",
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        default="smiles",
        help="Column name that stores SMILES strings.",
    )
    parser.add_argument(
        "--name_column",
        type=str,
        default="index",
        help="Column name used to build image file names.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Square image size in pixels.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="CSV chunk size used during streaming conversion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip image generation when the target PNG already exists.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="data/pretrain_imageGeneration_images/image_manifest.csv",
        help="CSV file used to record successfully generated images.",
    )
    parser.add_argument(
        "--invalid_csv",
        type=str,
        default="data/pretrain_imageGeneration_images/invalid_smiles.csv",
        help="CSV file used to record rows with invalid or empty SMILES.",
    )
    return parser.parse_args()


def sanitize_filename(raw_name: object, fallback: str) -> str:
    text = str(raw_name).strip() if raw_name is not None else ""
    if not text or text.lower() == "nan":
        text = fallback
    text = re.sub(r"[^\w.-]+", "_", text)
    text = text.strip("._")
    return text or fallback


def resolve_unique_image_path(output_dir: Path, base_name: str) -> Path:
    candidate = output_dir / f"{base_name}.png"
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = output_dir / f"{base_name}_{suffix}.png"
        if not candidate.exists():
            return candidate
        suffix += 1


def validate_columns(input_csv: Path, smiles_column: str, name_column: str) -> None:
    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = [col for col in [smiles_column, name_column] if col and col not in columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")


def convert_csv_to_images(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    manifest_csv = Path(args.manifest_csv)
    invalid_csv = Path(args.invalid_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    validate_columns(input_csv, args.smiles_column, args.name_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    invalid_csv.parent.mkdir(parents=True, exist_ok=True)

    usecols = [args.smiles_column, args.name_column]
    processed_rows = 0
    saved_rows = 0
    skipped_existing = 0
    invalid_rows = 0

    with manifest_csv.open("w", newline="", encoding="utf-8") as manifest_handle, invalid_csv.open(
        "w", newline="", encoding="utf-8"
    ) as invalid_handle:
        manifest_writer = csv.DictWriter(
            manifest_handle,
            fieldnames=["source_row", "source_name", "smiles", "image_name", "image_path"],
        )
        manifest_writer.writeheader()

        invalid_writer = csv.DictWriter(
            invalid_handle,
            fieldnames=["source_row", "source_name", "smiles", "reason"],
        )
        invalid_writer.writeheader()

        progress_total = args.limit if args.limit is not None else None
        progress = tqdm(total=progress_total, desc="Generating images", unit="mol")

        for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=args.chunksize):
            for row in chunk.itertuples(index=False):
                if args.limit is not None and processed_rows >= args.limit:
                    progress.close()
                    print(
                        f"Done. processed={processed_rows}, saved={saved_rows}, "
                        f"skipped_existing={skipped_existing}, invalid={invalid_rows}"
                    )
                    print(f"Images: {output_dir}")
                    print(f"Manifest: {manifest_csv}")
                    print(f"Invalid rows: {invalid_csv}")
                    return

                smiles = getattr(row, args.smiles_column)
                source_name = getattr(row, args.name_column)
                source_row = processed_rows + 1
                fallback_name = f"mol_{source_row:08d}"
                base_name = sanitize_filename(source_name, fallback_name)

                if pd.isna(smiles) or not str(smiles).strip():
                    invalid_writer.writerow(
                        {
                            "source_row": source_row,
                            "source_name": source_name,
                            "smiles": smiles,
                            "reason": "empty_smiles",
                        }
                    )
                    invalid_rows += 1
                    processed_rows += 1
                    progress.update(1)
                    continue

                smiles = str(smiles).strip()
                image_path = output_dir / f"{base_name}.png"

                if image_path.exists() and args.skip_existing:
                    manifest_writer.writerow(
                        {
                            "source_row": source_row,
                            "source_name": source_name,
                            "smiles": smiles,
                            "image_name": image_path.name,
                            "image_path": str(image_path),
                        }
                    )
                    skipped_existing += 1
                    processed_rows += 1
                    progress.update(1)
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    invalid_writer.writerow(
                        {
                            "source_row": source_row,
                            "source_name": source_name,
                            "smiles": smiles,
                            "reason": "invalid_smiles",
                        }
                    )
                    invalid_rows += 1
                    processed_rows += 1
                    progress.update(1)
                    continue

                if image_path.exists() and not args.skip_existing:
                    image_path = resolve_unique_image_path(output_dir, base_name)

                image = Draw.MolToImage(mol, size=(args.image_size, args.image_size), imageType="png")
                image.save(image_path)

                manifest_writer.writerow(
                    {
                        "source_row": source_row,
                        "source_name": source_name,
                        "smiles": smiles,
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                    }
                )

                saved_rows += 1
                processed_rows += 1
                progress.update(1)

        progress.close()

    print(
        f"Done. processed={processed_rows}, saved={saved_rows}, "
        f"skipped_existing={skipped_existing}, invalid={invalid_rows}"
    )
    print(f"Images: {output_dir}")
    print(f"Manifest: {manifest_csv}")
    print(f"Invalid rows: {invalid_csv}")


if __name__ == "__main__":
    convert_csv_to_images(parse_args())

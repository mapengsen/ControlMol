import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import select_best_decoupled_smiles as base


def _split_comma_values(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(", ") if item.strip()]


def _dedup_by_subfolder_path(df: pd.DataFrame, *, failed: bool) -> pd.DataFrame:
    """Keep a single highest-priority row for each subfolder_path + source_smiles pair. """
    if df.empty or "subfolder_path" not in df.columns:
        return df

    subset_cols = ["subfolder_path"]
    if "main_folder" in df.columns:
        subset_cols = ["main_folder", "subfolder_path"]
    if "source_smiles" in df.columns:
        subset_cols.append("source_smiles")

    if failed:
        sim_passed = df.get("similarity_passed")
        sim_passed_val = sim_passed.astype(float) if sim_passed is not None else 0.0
        similarity = pd.to_numeric(df.get("similarity"), errors="coerce").fillna(-np.inf)
        score = pd.to_numeric(df.get("decoupling_score"), errors="coerce").fillna(-np.inf)
        ordered = (
            df.assign(_passed=sim_passed_val, _sim=similarity, _score=score)
            .sort_values(
                by=["_passed", "_sim", "_score"],
                ascending=[False, False, False],
                kind="mergesort",
            )
        )
        deduped = ordered.drop_duplicates(subset=subset_cols, keep="first")
        return deduped.drop(columns=["_passed", "_sim", "_score"], errors="ignore").reset_index(drop=True)

    score = pd.to_numeric(df.get("decoupling_score"), errors="coerce").fillna(-np.inf)
    ratio_dev = pd.to_numeric(df.get("ratio_deviation"), errors="coerce").fillna(np.inf)
    change_rate = pd.to_numeric(df.get("target_change_rate"), errors="coerce").fillna(-np.inf)
    confidence = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(-np.inf)

    ordered = (
        df.assign(_score=score, _ratio_dev=ratio_dev, _change=change_rate, _conf=confidence)
        .sort_values(
            by=["_score", "_ratio_dev", "_change", "_conf"],
            ascending=[False, True, False, False],
            kind="mergesort",
            )
    )
    deduped = ordered.drop_duplicates(subset=subset_cols, keep="first")
    return deduped.drop(columns=["_score", "_ratio_dev", "_change", "_conf"], errors="ignore").reset_index(drop=True)


def _run_prompt_molopt_prediction_if_needed(
    df: pd.DataFrame,
    input_csv_path: Path,
    column_prefix_map: "Dict[str, str]",
    properties: List[str],
    property_suffix: str,
    prediction_script: Path,
    hyperparam_pickle: Path,
    model_names: Iterable[str],
    batch_size: int,
    num_workers: int,
    device: str,
    num_seeds: int,
    force: bool,
    skip: bool,
) -> Tuple[pd.DataFrame, bool]:
    prediction_performed = False
    if not column_prefix_map or not properties:
        return df, prediction_performed

    missing_targets = base._find_missing_prediction_columns(df, column_prefix_map, property_suffix, properties)
    need_prediction = force or any(missing_targets.values())
    if not need_prediction:
        return df, prediction_performed

    if skip:
        if missing_targets:
            for smiles_col, absent in missing_targets.items():
                print(f"Skipping property prediction: column {smiles_col} is missing {absent}")
        return df, prediction_performed

    if not prediction_script.exists():
        raise FileNotFoundError(f"Could not find the Prompt-MolOpt prediction script: {prediction_script}")
    if not hyperparam_pickle.exists():
        raise FileNotFoundError(f"Could not find the hyperparameter file: {hyperparam_pickle}")
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Could not find the input CSV file: {input_csv_path}")

    smiles_cols = list(column_prefix_map.keys())
    tasks = [t for t in model_names if t]
    if not tasks:
        tasks = properties

    with tempfile.NamedTemporaryFile(prefix="prompt_predictions_", suffix=".csv", delete=False) as tmp_file:
        temp_output_path = Path(tmp_file.name)

    command: List[str] = [
        sys.executable,
        str(prediction_script),
        "--hyperparam_pickle",
        str(hyperparam_pickle),
        "--input_csv",
        str(input_csv_path),
        "--output_csv",
        str(temp_output_path),
        "--smiles_columns",
        ", ".join(smiles_cols),
        "--model_names",
        ", ".join(tasks),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--device",
        device,
        "--num_seeds",
        str(num_seeds),
    ]

    print("[Property Prediction] Running Prompt-MolOpt predict_ADMET_pro.py")

    try:
        subprocess.run(command, check=True)
        predictions_df = pd.read_csv(temp_output_path)
        if len(predictions_df) != len(df):
            raise ValueError("The prediction output row count does not match the input CSV. Please check the prediction script output. ")

        for smiles_col, prefix in column_prefix_map.items():
            raw_prefix = f"{smiles_col}_"
            for prop in properties:
                raw_col = f"{raw_prefix}{prop}_pred"
                target_col = f"{prefix}_{prop}{property_suffix}"
                if raw_col not in predictions_df.columns:
                    print(f"Warning: prediction output is missing column {raw_col}; skipping write to {target_col}")
                    continue
                df[target_col] = predictions_df[raw_col].values

        prediction_performed = True
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Prompt-MolOpt property prediction failed: {exc}") from exc
    finally:
        try:
            temp_output_path.unlink()
        except FileNotFoundError:
            pass

    return df, prediction_performed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best decoupled molecules using Prompt-MolOpt property prediction and output the best SMILES. "
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the detailed candidate CSV. ",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to save the selected best molecules. ",
    )
    parser.add_argument(
        "--all_success_output_csv",
        default=None,
        help="Optional CSV path for all successful decoupled candidates. ",
    )
    parser.add_argument(
        "--target_property",
        default="BBBP",
        help="Target property optimized in this run (default: BBBP). ",
    )
    parser.add_argument(
        "--properties_to_preserve",
        default=", ".join(base.DEFAULT_PROPERTIES_TO_PRESERVE),
        help="Comma-separated list of properties that should remain stable. ",
    )
    parser.add_argument(
        "--optimized_smiles_column",
        default="smiles",
        help="Column name containing optimized SMILES. ",
    )
    parser.add_argument(
        "--source_smiles_column",
        default="source_smiles",
        help="Column name containing source SMILES. ",
    )
    parser.add_argument(
        "--optimized_property_prefix",
        default=None,
        help="Prefix used for optimized property columns. Defaults to optimized_smiles_column. ",
    )
    parser.add_argument(
        "--source_property_prefix",
        default=None,
        help="Prefix used for source property columns. Defaults to source_smiles_column. ",
    )
    parser.add_argument(
        "--property_suffix",
        default="_pred",
        help="Suffix for property columns, for example `_pred`. Leave empty to disable the suffix. ",
    )
    parser.add_argument(
        "--prediction_script",
        default="predictModel/Prompt-MolOpt/sme_opt_utils/predict_ADMET_pro.py",
        help="Path to the Prompt-MolOpt prediction script. ",
    )
    parser.add_argument(
        "--prediction_hyperparam_pickle",
        default="predictModel/Prompt-MolOpt/checkpoints/hyperparameter_ADMET_data_for_MGA. pkl",
        help="Path to the Prompt-MolOpt hyperparameter pickle file. ",
    )
    parser.add_argument(
        "--prediction_model_names",
        default=None,
        help="Comma-separated prediction task names. Defaults to target_property plus properties_to_preserve. ",
    )
    parser.add_argument(
        "--prediction_batch_size",
        type=int,
        default=1024,
        help="Batch size for Prompt-MolOpt property prediction. ",
    )
    parser.add_argument(
        "--prediction_num_workers",
        type=int,
        default=16,
        help="Number of DataLoader workers for Prompt-MolOpt property prediction. ",
    )
    parser.add_argument(
        "--prediction_device",
        default="cuda",
        help="Device used for property prediction, for example cuda or cpu. ",
    )
    parser.add_argument(
        "--prediction_num_seeds",
        type=int,
        default=10,
        help="Number of Prompt-MolOpt ensemble seeds/checkpoints to use. ",
    )
    parser.add_argument(
        "--skip_property_prediction",
        action="store_true",
        help="Skip the property prediction step. ",
    )
    parser.add_argument(
        "--force_property_prediction",
        action="store_true",
        help="Force property prediction even if prediction columns already exist. ",
    )
    parser.add_argument(
        "--similarity_column",
        default="similarity",
        help="Name of the similarity column in the CSV. Leave empty to compute it on the fly. ",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.4,
        help="Minimum Tanimoto similarity required when filtering candidates. ",
    )
    parser.add_argument(
        "--similarity_fingerprint",
        choices=base.FINGERPRINT_TYPES,
        default=base.DEFAULT_FINGERPRINT_TYPE,
        help="Fingerprint type used to compute Tanimoto similarity: morgan or maccs. ",
    )
    parser.add_argument(
        "--failed_output_csv",
        default=None,
        help="Fallback CSV path used when no successful decoupled candidate is found. Defaults to output_csv with a suffix. ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if df.empty:
        print("Input CSV is empty; nothing to process. ")
        return

    properties_to_preserve = _split_comma_values(args.properties_to_preserve)
    if not properties_to_preserve:
        print("No properties to preserve were provided; cannot compute decoupling scores. ")
        return
    all_properties = list(dict.fromkeys([args.target_property, *properties_to_preserve]))

    column_prefix_map = base._collect_prediction_column_map(
        args.optimized_smiles_column,
        args.optimized_property_prefix,
        args.source_smiles_column,
        args.source_property_prefix,
        None,
    )

    model_names = _split_comma_values(args.prediction_model_names) if args.prediction_model_names else all_properties
    df, prediction_performed = _run_prompt_molopt_prediction_if_needed(
        df=df,
        input_csv_path=Path(args.input_csv),
        column_prefix_map=column_prefix_map,
        properties=all_properties,
        property_suffix=args.property_suffix or "",
        prediction_script=Path(args.prediction_script),
        hyperparam_pickle=Path(args.prediction_hyperparam_pickle),
        model_names=model_names,
        batch_size=args.prediction_batch_size,
        num_workers=args.prediction_num_workers,
        device=args.prediction_device,
        num_seeds=args.prediction_num_seeds,
        force=args.force_property_prediction,
        skip=args.skip_property_prediction,
    )

    if prediction_performed:
        try:
            df.to_csv(args.input_csv, index=False)
            print(f"Saved the property-augmented input CSV to {args. input_csv}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: could not save the property-augmented input CSV {args. input_csv}: {exc}")

    collect_all_success = bool(args.all_success_output_csv)
    process_result = base.process_groups(
        df,
        target_property=args.target_property,
        properties_to_preserve=properties_to_preserve,
        property_suffix=args.property_suffix or "",
        optimized_smiles_column=args.optimized_smiles_column,
        source_smiles_column=args.source_smiles_column,
        optimized_property_prefix=args.optimized_property_prefix,
        source_property_prefix=args.source_property_prefix,
        similarity_column=args.similarity_column,
        min_similarity=args.min_similarity,
        fingerprint_type=args.similarity_fingerprint,
        collect_all_success=collect_all_success,
    )
    if collect_all_success:
        success_df, failed_df, all_success_df = process_result
    else:
        success_df, failed_df = process_result
        all_success_df = pd.DataFrame()

    # Keep one preferred record per subfolder after selection.
    success_df = _dedup_by_subfolder_path(success_df, failed=False)
    failed_df = _dedup_by_subfolder_path(failed_df, failed=True)

    success_count = len(success_df.index)
    denominator = None
    if args.source_smiles_column and args.source_smiles_column in df.columns:
        denominator = df[args.source_smiles_column].dropna().nunique()
    elif "main_folder" in df.columns:
        denominator = df["main_folder"].dropna().nunique()

    if denominator:
        success_rate = success_count / denominator
        print(f"Decoupling success rate: {success_count}/{denominator} = {success_rate: .2%}")
    else:
        print("Warning: could not determine the denominator for success rate (missing source_smiles or main_folder columns). ")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if success_df.empty:
        print("No optimized results passed selection. ")
        success_df.to_csv(output_path, index=False)
        print(f"Saved empty result file to {output_path}")
    else:
        success_df.to_csv(output_path, index=False)
        print(f"Saved selected results to {output_path}")

    if collect_all_success:
        all_success_path = Path(args.all_success_output_csv)
        all_success_path.parent.mkdir(parents=True, exist_ok=True)
        all_success_df.to_csv(all_success_path, index=False)
        if all_success_df.empty:
            print(f"All successful decoupled results are empty; wrote an empty CSV to {all_success_path}")
        else:
            print(f"Saved all successful decoupled candidates to {all_success_path}")

    combined_df = pd.concat([success_df, failed_df], ignore_index=True, sort=False)
    failed_output_path = Path(args.failed_output_csv) if args.failed_output_csv else base._default_failed_output(output_path)
    if not combined_df.empty:
        failed_output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(failed_output_path, index=False)
        print(f"Saved candidates (successful + fallback) to {failed_output_path}")
    else:
        print("No candidates found (neither successful nor fallback). ")


if __name__ == "__main__":
    main()

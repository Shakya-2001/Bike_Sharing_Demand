from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.generate_data_errors import inject_errors
from scripts.split_dataset import DEFAULT_COLUMNS, ROWS_PER_FILE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo_data" / "defense1"
DEMO_FILE_NAMES = [
    "01_all_invalid.csv",
    "02_clean.csv",
    "03_clean.csv",
    "04_clean.csv",
    "05_mixed.csv",
]


def prepare_demo_files(
    dataset_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> list[Path]:
    source = pd.read_csv(dataset_path)
    missing_columns = [
        column for column in DEFAULT_COLUMNS if column not in source.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    required_rows = len(DEMO_FILE_NAMES) * ROWS_PER_FILE
    if len(source) < required_rows:
        raise ValueError(
            f"Dataset must contain at least {required_rows} rows"
        )

    selected = source[DEFAULT_COLUMNS].sample(
        n=required_rows,
        random_state=seed,
    ).reset_index(drop=True)
    batches = [
        selected.iloc[
            index * ROWS_PER_FILE:(index + 1) * ROWS_PER_FILE
        ].copy()
        for index in range(len(DEMO_FILE_NAMES))
    ]

    batches[0] = inject_errors(
        batches[0],
        null_probability=1,
        range_probability=1,
        category_probability=1,
        type_probability=1,
        duplicate_probability=1,
        inconsistency_probability=1,
        outlier_probability=1,
        schema_probability=1,
        seed=seed,
    )

    mixed = batches[4]
    mixed.loc[mixed.index[0], "atemp"] = pd.NA
    mixed.loc[mixed.index[1], "hum"] = 2.5
    mixed["windspeed"] = mixed["windspeed"].astype("object")
    mixed.loc[mixed.index[2], "windspeed"] = "not_a_number"

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for file_name, batch in zip(DEMO_FILE_NAMES, batches):
        output_path = output_dir / file_name
        batch.to_csv(output_path, index=False)
        generated_files.append(output_path)

    manifest = {
        "rows_per_file": ROWS_PER_FILE,
        "files": {
            "01_all_invalid.csv": {
                "expected_criticality": "high",
                "expected_good_rows": 0,
                "expected_bad_rows": 10,
                "expected_alert": True,
            },
            "02_clean.csv": {
                "expected_criticality": "none",
                "expected_good_rows": 10,
                "expected_bad_rows": 0,
                "expected_alert": False,
            },
            "03_clean.csv": {
                "expected_criticality": "none",
                "expected_good_rows": 10,
                "expected_bad_rows": 0,
                "expected_alert": False,
            },
            "04_clean.csv": {
                "expected_criticality": "none",
                "expected_good_rows": 10,
                "expected_bad_rows": 0,
                "expected_alert": False,
            },
            "05_mixed.csv": {
                "expected_criticality": "medium",
                "expected_good_rows": 7,
                "expected_bad_rows": 3,
                "expected_alert": True,
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the five required Defense 1 demo files."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = prepare_demo_files(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Prepared {len(generated)} demo files in {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
ROWS_PER_FILE = 10


def split_dataset(
    dataset_path: Path,
    output_dir: Path,
    number_of_files: int,
    seed: int = 42,
) -> list[Path]:
    if number_of_files < 1:
        raise ValueError("number_of_files must be at least 1")

    df = pd.read_csv(dataset_path)
    missing_columns = [
        column for column in DEFAULT_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    required_rows = number_of_files * ROWS_PER_FILE
    if len(df) < required_rows:
        raise ValueError(
            f"Dataset has {len(df)} rows, but {required_rows} rows are "
            f"required to create {number_of_files} files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_df = df[DEFAULT_COLUMNS].sample(n=required_rows, random_state=seed)

    generated_files = []
    for index in range(number_of_files):
        start = index * ROWS_PER_FILE
        end = start + ROWS_PER_FILE
        batch = sampled_df.iloc[start:end]
        output_path = output_dir / f"batch_{index + 1:03d}.csv"
        batch.to_csv(output_path, index=False)
        generated_files.append(output_path)

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a dataset into ingestion files of exactly 10 rows each."
        )
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("number_of_files", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_files = split_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        number_of_files=args.number_of_files,
        seed=args.seed,
    )
    print(f"Generated {len(generated_files)} files in {args.output_dir}")


if __name__ == "__main__":
    main()

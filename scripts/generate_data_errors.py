from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42


def apply_row_error(
    df: pd.DataFrame,
    probability: float,
    column: str,
    value,
    seed_offset: int,
) -> pd.DataFrame:
    if column not in df.columns:
        return df
    if isinstance(value, str):
        df[column] = df[column].astype("object")
    mask = df.sample(
        frac=probability,
        random_state=RANDOM_SEED + seed_offset,
    ).index
    df.loc[mask, column] = value
    return df


def inject_errors(
    df: pd.DataFrame,
    null_probability: float = 0.05,
    range_probability: float = 0.05,
    category_probability: float = 0.05,
    type_probability: float = 0.05,
    duplicate_probability: float = 0.05,
    inconsistency_probability: float = 0.05,
    outlier_probability: float = 0.05,
    drop_column: bool = False,
) -> pd.DataFrame:
    corrupted = df.copy()

    corrupted = apply_row_error(corrupted, null_probability, "atemp", pd.NA, 1)
    corrupted = apply_row_error(corrupted, range_probability, "hum", 2.5, 2)
    corrupted = apply_row_error(
        corrupted, category_probability, "weathersit", 99, 3
    )
    corrupted = apply_row_error(
        corrupted, type_probability, "windspeed", "not_a_number", 4
    )
    corrupted = apply_row_error(
        corrupted, inconsistency_probability, "workingday", 1, 5
    )

    if {"workingday", "holiday"}.issubset(corrupted.columns):
        inconsistent_rows = corrupted.sample(
            frac=inconsistency_probability,
            random_state=RANDOM_SEED + 6,
        ).index
        corrupted.loc[inconsistent_rows, "holiday"] = 1
        corrupted.loc[inconsistent_rows, "workingday"] = 1

    corrupted = apply_row_error(
        corrupted, outlier_probability, "temp", 9.99, 7
    )

    if "instant" in corrupted.columns and duplicate_probability > 0:
        duplicate_rows = corrupted.sample(
            frac=duplicate_probability,
            random_state=RANDOM_SEED + 8,
        )
        corrupted = pd.concat([corrupted, duplicate_rows], ignore_index=True)

    if drop_column and "dteday" in corrupted.columns:
        corrupted = corrupted.drop(columns=["dteday"])

    return corrupted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject configurable data quality errors into a bike-sharing "
            "CSV file."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--null-probability", type=float, default=0.05)
    parser.add_argument("--range-probability", type=float, default=0.05)
    parser.add_argument("--category-probability", type=float, default=0.05)
    parser.add_argument("--type-probability", type=float, default=0.05)
    parser.add_argument("--duplicate-probability", type=float, default=0.05)
    parser.add_argument(
        "--inconsistency-probability",
        type=float,
        default=0.05,
    )
    parser.add_argument("--outlier-probability", type=float, default=0.05)
    parser.add_argument("--drop-column", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)
    corrupted = inject_errors(
        df,
        null_probability=args.null_probability,
        range_probability=args.range_probability,
        category_probability=args.category_probability,
        type_probability=args.type_probability,
        duplicate_probability=args.duplicate_probability,
        inconsistency_probability=args.inconsistency_probability,
        outlier_probability=args.outlier_probability,
        drop_column=args.drop_column,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    corrupted.to_csv(args.output_path, index=False)
    print(f"Saved corrupted data to {args.output_path}")


if __name__ == "__main__":
    main()

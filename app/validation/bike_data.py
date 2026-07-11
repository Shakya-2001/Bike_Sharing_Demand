from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.self_check.util import build_in_memory_runtime_context
from great_expectations.validator.validator import Validator


REQUIRED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "instant",
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

RANGE_RULES = {
    "season": (1, 4),
    "yr": (0, 1),
    "mnth": (1, 12),
    "hr": (0, 23),
    "holiday": (0, 1),
    "weekday": (0, 6),
    "workingday": (0, 1),
    "weathersit": (1, 4),
    "temp": (0, 1),
    "atemp": (0, 1),
    "hum": (0, 1),
    "windspeed": (0, 1),
}


@dataclass
class ValidationSummary:
    file_path: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    invalid_percent: float
    criticality: str
    missing_columns: list[str]
    error_counts: dict[str, int]
    gx_success: bool
    gx_failed_expectations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_validator(df: pd.DataFrame) -> Validator:
    return Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[Batch(data=df)],
        data_context=build_in_memory_runtime_context(),
    )


def run_gx_expectations(df: pd.DataFrame) -> tuple[bool, int]:
    gx_df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in gx_df.columns:
            gx_df[column] = pd.to_numeric(gx_df[column], errors="coerce")

    validator = build_validator(gx_df)

    for column in REQUIRED_COLUMNS:
        validator.expect_column_to_exist(column)

    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            validator.expect_column_values_to_not_be_null(column)

    for column, (min_value, max_value) in RANGE_RULES.items():
        if column in df.columns:
            validator.expect_column_values_to_be_between(
                column,
                min_value=min_value,
                max_value=max_value,
                mostly=1.0,
            )

    if "instant" in df.columns:
        validator.expect_column_values_to_be_unique("instant")

    result = validator.validate()
    failed_expectations = sum(1 for item in result.results if not item.success)
    return bool(result.success), failed_expectations


def collect_invalid_row_indices(
    df: pd.DataFrame,
) -> tuple[set[int], dict[str, int]]:
    invalid_indices: set[int] = set()
    error_counts = {
        "null_required_value": 0,
        "invalid_range": 0,
        "invalid_numeric_type": 0,
        "duplicate_instant": 0,
        "holiday_workingday_inconsistency": 0,
    }

    present_required_columns = [
        column for column in REQUIRED_COLUMNS if column in df.columns
    ]
    for index in df[df[present_required_columns].isna().any(axis=1)].index:
        invalid_indices.add(int(index))
        error_counts["null_required_value"] += 1

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        invalid_type_mask = numeric_values.isna() & df[column].notna()
        bad_type_indices = df[invalid_type_mask].index
        for index in bad_type_indices:
            invalid_indices.add(int(index))
        error_counts["invalid_numeric_type"] += len(bad_type_indices)

    for column, (min_value, max_value) in RANGE_RULES.items():
        if column not in df.columns:
            continue
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        invalid_range_mask = (
            (numeric_values < min_value) | (numeric_values > max_value)
        )
        bad_range_indices = df[invalid_range_mask.fillna(False)].index
        for index in bad_range_indices:
            invalid_indices.add(int(index))
        error_counts["invalid_range"] += len(bad_range_indices)

    if "instant" in df.columns:
        duplicate_indices = df[df["instant"].duplicated(keep=False)].index
        for index in duplicate_indices:
            invalid_indices.add(int(index))
        error_counts["duplicate_instant"] += len(duplicate_indices)

    if {"holiday", "workingday"}.issubset(df.columns):
        holiday = pd.to_numeric(df["holiday"], errors="coerce")
        workingday = pd.to_numeric(df["workingday"], errors="coerce")
        inconsistent_indices = df[(holiday == 1) & (workingday == 1)].index
        for index in inconsistent_indices:
            invalid_indices.add(int(index))
        error_counts["holiday_workingday_inconsistency"] += len(
            inconsistent_indices
        )

    return invalid_indices, error_counts


def determine_criticality(
    missing_columns: list[str],
    invalid_percent: float,
) -> str:
    if missing_columns or invalid_percent > 50:
        return "high"
    if 10 <= invalid_percent <= 50:
        return "medium"
    if 0 < invalid_percent < 10:
        return "low"
    return "none"


def validate_bike_data(
    file_path: Path,
) -> tuple[ValidationSummary, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    gx_success, gx_failed_expectations = run_gx_expectations(df)
    invalid_indices, error_counts = collect_invalid_row_indices(df)

    invalid_rows = len(invalid_indices)
    total_rows = len(df)
    invalid_percent = (invalid_rows / total_rows * 100) if total_rows else 0.0
    criticality = determine_criticality(missing_columns, invalid_percent)

    if invalid_indices:
        bad_df = df.loc[sorted(invalid_indices)].copy()
        good_df = df.drop(index=sorted(invalid_indices)).copy()
    else:
        bad_df = df.iloc[0:0].copy()
        good_df = df.copy()

    summary = ValidationSummary(
        file_path=str(file_path),
        total_rows=total_rows,
        valid_rows=len(good_df),
        invalid_rows=invalid_rows,
        invalid_percent=round(invalid_percent, 2),
        criticality=criticality,
        missing_columns=missing_columns,
        error_counts=error_counts,
        gx_success=gx_success,
        gx_failed_expectations=gx_failed_expectations,
    )
    return summary, good_df, bad_df

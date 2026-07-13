from __future__ import annotations

import pandas as pd
import pytest

from scripts.generate_data_errors import inject_errors


def clean_bike_rows(row_count: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instant": index + 1,
                "dteday": "2012-01-01",
                "season": 1,
                "yr": 1,
                "mnth": 1,
                "hr": index % 24,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.5,
                "atemp": 0.5,
                "hum": 0.5,
                "windspeed": 0.2,
            }
            for index in range(row_count)
        ]
    )


def test_inject_errors_can_create_required_error_types() -> None:
    df = clean_bike_rows()

    corrupted = inject_errors(
        df,
        null_probability=0.2,
        range_probability=0.2,
        category_probability=0.2,
        type_probability=0.2,
        duplicate_probability=0.2,
        inconsistency_probability=0.2,
        outlier_probability=0.2,
        drop_column=True,
        seed=42,
    )

    assert len(corrupted) == len(df)
    assert "dteday" not in corrupted.columns
    assert corrupted["atemp"].isna().sum() > 0
    assert (corrupted["temp"] == 9.99).sum() > 0
    assert (corrupted["hum"] == 2.5).sum() > 0
    assert (corrupted["weathersit"] == 99).sum() > 0
    assert (corrupted["windspeed"] == "not_a_number").sum() > 0
    assert corrupted["instant"].duplicated().sum() > 0
    inconsistent_rows = (
        (corrupted["holiday"] == 1) & (corrupted["workingday"] == 1)
    )
    assert inconsistent_rows.sum() > 0


def test_positive_probabilities_work_for_ten_row_demo_file() -> None:
    df = clean_bike_rows(row_count=10)

    corrupted = inject_errors(
        df,
        schema_probability=0,
        seed=7,
    )

    assert len(corrupted) == 10
    assert corrupted["atemp"].isna().sum() >= 1
    assert (corrupted["hum"] == 2.5).sum() >= 1
    assert (corrupted["weathersit"] == 99).sum() >= 1
    assert (corrupted["windspeed"] == "not_a_number").sum() >= 1
    assert (corrupted["temp"] == 9.99).sum() >= 1
    assert corrupted["instant"].duplicated(keep=False).sum() >= 2
    inconsistent_rows = (
        (corrupted["holiday"] == 1) & (corrupted["workingday"] == 1)
    )
    assert inconsistent_rows.sum() >= 1


def test_seed_makes_corruption_reproducible() -> None:
    df = clean_bike_rows()

    first = inject_errors(df, schema_probability=0, seed=123)
    second = inject_errors(df, schema_probability=0, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_schema_probability_can_remove_required_column() -> None:
    corrupted = inject_errors(
        clean_bike_rows(row_count=10),
        schema_probability=1,
        seed=1,
    )

    assert "dteday" not in corrupted.columns


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="null_probability"):
        inject_errors(clean_bike_rows(), null_probability=1.1)

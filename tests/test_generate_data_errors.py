from __future__ import annotations

import pandas as pd

from scripts.generate_data_errors import inject_errors


def test_inject_errors_can_create_required_error_types() -> None:
    df = pd.DataFrame(
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
            for index in range(20)
        ]
    )

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
    )

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

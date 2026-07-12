from __future__ import annotations

from typing import Any

import pandas as pd


FEATURE_COLUMNS = [
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

SEASON_MONTHS = {
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11],
    4: [12, 1, 2],
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

WEEKDAY_NAMES = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

SOURCE_NAMES = {
    "all": "All predictions",
    "webapp": "Manual predictions",
    "scheduled": "Scheduled predictions",
}


def normalize_temperature(celsius: float) -> float:
    return (celsius - (-8.0)) / (39.0 - (-8.0))


def normalize_feels_like_temperature(celsius: float) -> float:
    return (celsius - (-16.0)) / (50.0 - (-16.0))


def normalize_humidity(percent: int) -> float:
    return percent / 100.0


def normalize_wind_speed(km_per_hour: float) -> float:
    return km_per_hour / 67.0


def batch_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    missing = [column for column in FEATURE_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    if data[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Feature values cannot be empty.")
    return data[FEATURE_COLUMNS].to_dict(orient="records")


def prediction_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("predictions", []):
        row = dict(item.get("features", {}))
        row["prediction"] = item.get("prediction")
        rows.append(row)
    return pd.DataFrame(rows, columns=[*FEATURE_COLUMNS, "prediction"])


def history_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table_rows = []
    for item in rows:
        row = dict(item.get("features", {}))
        row.update(
            {
                "prediction": item.get("prediction"),
                "source": item.get("source"),
                "model_name": item.get("model_name"),
                "created_at": item.get("created_at"),
            }
        )
        table_rows.append(row)
    return pd.DataFrame(
        table_rows,
        columns=[
            "created_at",
            "source",
            "model_name",
            *FEATURE_COLUMNS,
            "prediction",
        ],
    )

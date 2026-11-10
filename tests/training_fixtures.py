from __future__ import annotations

import pandas as pd


def make_dataset(rows: int = 100) -> pd.DataFrame:
    """A small fake hour.csv with the columns the training code needs."""
    return pd.DataFrame(
        {
            "instant": list(range(1, rows + 1)),
            "dteday": "2012-01-01",
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": [index % 24 for index in range(rows)],
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
            "cnt": [10 + index for index in range(rows)],
        }
    )


def write_good_file(good_dir, name: str, dataset: pd.DataFrame, instants):
    """Write an ingested file (features only, no cnt) for some rows."""
    good_dir.mkdir(parents=True, exist_ok=True)
    rows = dataset[dataset["instant"].isin(instants)].drop(columns=["cnt"])
    rows.to_csv(good_dir / name, index=False)
    return good_dir / name


class FixedModel:
    """A fake model that always predicts the same number."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame) -> list[float]:
        return [self.value] * len(frame)


class FakeVersion:
    def __init__(self, version: str) -> None:
        self.version = version


class FakeHook:
    """Records the SQL that would be sent to PostgreSQL."""

    def __init__(self, baseline_count: int = 3) -> None:
        self.runs: list[tuple] = []
        self.inserted: list[dict] = []
        self.baseline_count = baseline_count

    def run(self, sql, parameters=None) -> None:
        self.runs.append((sql, parameters))

    def insert_rows(self, table, rows, target_fields, commit_every) -> None:
        self.inserted.append(
            {"table": table, "rows": rows, "fields": target_fields}
        )

    def get_first(self, sql, parameters=None):
        return (self.baseline_count,)

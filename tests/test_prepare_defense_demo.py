from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.validation.bike_data import (
    REQUIRED_COLUMNS,
    collect_invalid_row_indices,
    determine_criticality,
)
from scripts.prepare_defense_demo import DEMO_FILE_NAMES, prepare_demo_files


def source_dataset(row_count: int = 60) -> pd.DataFrame:
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


def validation_counts(path: Path) -> tuple[str, int, int]:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    invalid_indices, _ = collect_invalid_row_indices(frame, missing)
    invalid_rows = len(invalid_indices)
    invalid_percent = invalid_rows / len(frame) * 100
    return (
        determine_criticality(missing, invalid_percent),
        len(frame) - invalid_rows,
        invalid_rows,
    )


def test_prepare_demo_files_creates_expected_scenarios(tmp_path: Path) -> None:
    dataset_path = tmp_path / "hour.csv"
    output_dir = tmp_path / "demo"
    source_dataset().to_csv(dataset_path, index=False)

    generated = prepare_demo_files(dataset_path, output_dir, seed=42)

    assert [path.name for path in generated] == DEMO_FILE_NAMES
    assert all(len(pd.read_csv(path)) == 10 for path in generated)
    assert validation_counts(generated[0]) == ("high", 0, 10)
    assert validation_counts(generated[1]) == ("none", 10, 0)
    assert validation_counts(generated[2]) == ("none", 10, 0)
    assert validation_counts(generated[3]) == ("none", 10, 0)
    assert validation_counts(generated[4]) == ("medium", 7, 3)

    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["rows_per_file"] == 10
    assert manifest["files"]["01_all_invalid.csv"][
        "expected_alert"
    ] is True


def test_prepare_demo_files_rejects_missing_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "hour.csv"
    frame = source_dataset().drop(columns=["dteday"])
    frame.to_csv(dataset_path, index=False)

    try:
        prepare_demo_files(dataset_path, tmp_path / "demo")
    except ValueError as exc:
        assert "dteday" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation to fail")

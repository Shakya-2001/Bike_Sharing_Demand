from __future__ import annotations

import pandas as pd

from app.ingestion.pipeline import ingest_file


def make_rows(count: int = 10) -> list[dict]:
    return [
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
        for index in range(count)
    ]


def test_ingest_file_moves_clean_file_to_good_data(tmp_path) -> None:
    raw_file = tmp_path / "raw" / "batch.csv"
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    raw_file.parent.mkdir()
    pd.DataFrame(make_rows()).to_csv(raw_file, index=False)

    summary, outputs = ingest_file(raw_file, good_dir, bad_dir)

    assert summary.criticality == "none"
    assert outputs["good_file"] is not None
    assert outputs["bad_file"] is None
    assert not raw_file.exists()
    assert len(list(good_dir.glob("*.csv"))) == 1


def test_ingest_file_splits_mixed_file(tmp_path) -> None:
    raw_file = tmp_path / "raw" / "mixed.csv"
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    raw_file.parent.mkdir()
    rows = make_rows()
    rows[0]["hum"] = 2.5
    rows[1]["weathersit"] = 99
    pd.DataFrame(rows).to_csv(raw_file, index=False)

    summary, outputs = ingest_file(raw_file, good_dir, bad_dir)

    assert summary.criticality == "medium"
    assert outputs["good_file"] is not None
    assert outputs["bad_file"] is not None
    assert not raw_file.exists()
    assert len(pd.read_csv(outputs["good_file"])) == 8
    assert len(pd.read_csv(outputs["bad_file"])) == 2

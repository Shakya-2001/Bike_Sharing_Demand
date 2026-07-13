from __future__ import annotations

import json

import pandas as pd
import pytest

from dags import prediction_dag


def make_rows(count: int = 2) -> list[dict]:
    return [
        {
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": index,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
            "cnt": 10,
        }
        for index in range(count)
    ]


def test_prediction_dag_metadata() -> None:
    assert prediction_dag.DAG_ID == "bike_demand_prediction"
    assert prediction_dag.TASK_IDS == [
        "check_for_new_data",
        "make_predictions",
    ]


def test_find_new_files_excludes_processed_files(tmp_path) -> None:
    good_dir = tmp_path / "good_data"
    state_file = tmp_path / "state" / "processed_files.json"
    good_dir.mkdir()
    first = good_dir / "batch_001.csv"
    second = good_dir / "batch_002.csv"
    pd.DataFrame(make_rows()).to_csv(first, index=False)
    pd.DataFrame(make_rows()).to_csv(second, index=False)
    state_file.parent.mkdir()
    state_file.write_text(
        json.dumps({"processed_files": [str(first.resolve())]}),
        encoding="utf-8",
    )

    files = prediction_dag.find_new_good_files(
        str(good_dir),
        str(state_file),
    )

    assert files == [str(second)]


def test_check_skips_when_no_new_data(tmp_path) -> None:
    config = {
        "good_data_dir": str(tmp_path / "good_data"),
        "state_file": str(tmp_path / "state.json"),
        "api_url": "http://api:8000",
    }

    with pytest.raises(prediction_dag.AirflowSkipException):
        prediction_dag.check_for_new_data(config)


def test_batch_prediction_makes_one_api_call(monkeypatch, tmp_path) -> None:
    good_dir = tmp_path / "good_data"
    state_file = tmp_path / "state" / "processed_files.json"
    good_dir.mkdir()
    first = good_dir / "batch_001.csv"
    second = good_dir / "batch_002.csv"
    pd.DataFrame(make_rows(2)).to_csv(first, index=False)
    pd.DataFrame(make_rows(3)).to_csv(second, index=False)
    calls = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model_name": "bike-model-v1",
                "predictions": [
                    {"prediction": 42.0, "features": row}
                    for row in calls[0]["json"]["features"]
                ],
            }

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(prediction_dag.requests, "post", fake_post)
    payload = {
        "files": [str(first), str(second)],
        "state_file": str(state_file),
        "api_url": "http://api:8000",
    }

    result = prediction_dag.make_batch_predictions(payload)

    assert len(calls) == 1
    assert calls[0]["url"] == "http://api:8000/predict"
    assert calls[0]["json"]["source"] == "scheduled"
    assert len(calls[0]["json"]["features"]) == 5
    assert "cnt" not in calls[0]["json"]["features"][0]
    assert result["files_processed"] == 2
    assert result["rows_predicted"] == 5
    assert first.exists()
    assert second.exists()
    processed = prediction_dag.read_processed_files(state_file)
    assert processed == {str(first.resolve()), str(second.resolve())}


def test_failed_api_call_does_not_mark_files_processed(
    monkeypatch,
    tmp_path,
) -> None:
    good_dir = tmp_path / "good_data"
    state_file = tmp_path / "state" / "processed_files.json"
    good_dir.mkdir()
    file_path = good_dir / "batch_001.csv"
    pd.DataFrame(make_rows()).to_csv(file_path, index=False)

    def fail_post(url, json, timeout):
        raise prediction_dag.requests.ConnectionError("API unavailable")

    monkeypatch.setattr(prediction_dag.requests, "post", fail_post)
    payload = {
        "files": [str(file_path)],
        "state_file": str(state_file),
        "api_url": "http://api:8000",
    }

    with pytest.raises(prediction_dag.requests.ConnectionError):
        prediction_dag.make_batch_predictions(payload)

    assert not state_file.exists()
    assert file_path.exists()

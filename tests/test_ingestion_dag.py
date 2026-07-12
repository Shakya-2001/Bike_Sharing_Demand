from __future__ import annotations

from pathlib import Path

import pandas as pd

from dags import ingestion_dag
from scripts.split_dataset import DEFAULT_COLUMNS


def make_rows(count: int = 20) -> list[dict]:
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
            "casual": 1,
            "registered": 2,
            "cnt": 3,
        }
        for index in range(count)
    ]


def test_ingestion_dag_metadata() -> None:
    assert ingestion_dag.DAG_ID == "bike_data_ingestion"
    assert ingestion_dag.TASK_IDS == [
        "split_dataset",
        "read_data",
        "validate_data",
        "save_statistics",
        "send_alerts",
        "split_and_save_data",
    ]


def test_create_raw_data_files_generates_batches(tmp_path) -> None:
    dataset_path = tmp_path / "hour.csv"
    raw_data_dir = tmp_path / "raw_data"
    pd.DataFrame(make_rows()).to_csv(dataset_path, index=False)

    config = {
        "dataset_path": str(dataset_path),
        "raw_data_dir": str(raw_data_dir),
        "good_data_dir": str(tmp_path / "good_data"),
        "bad_data_dir": str(tmp_path / "bad_data"),
        "number_of_files": 2,
    }

    files = ingestion_dag.create_raw_data_files(config)

    assert len(files) == 2
    for file_path in files:
        batch = pd.read_csv(file_path)
        assert len(batch) == 10
        assert list(batch.columns) == DEFAULT_COLUMNS


def test_validate_and_split_file(tmp_path) -> None:
    raw_data_dir = tmp_path / "raw_data"
    good_data_dir = tmp_path / "good_data"
    bad_data_dir = tmp_path / "bad_data"
    raw_data_dir.mkdir()
    file_path = raw_data_dir / "batch_001.csv"
    pd.DataFrame(make_rows(10)).to_csv(file_path, index=False)

    selected_file = ingestion_dag.read_random_raw_file(str(raw_data_dir))
    validation_result = ingestion_dag.validate_file(selected_file)
    outputs = ingestion_dag.split_valid_and_invalid_data(
        validation_result=validation_result,
        good_data_dir=str(good_data_dir),
        bad_data_dir=str(bad_data_dir),
    )

    assert validation_result["status"] == "validated"
    assert validation_result["summary"]["criticality"] == "none"
    assert Path(outputs["good_file"]).exists()
    assert outputs["bad_file"] is None
    assert not file_path.exists()


def test_build_alert_message_for_high_criticality() -> None:
    validation_result = {
        "status": "validated",
        "summary": {
            "file_path": "batch_999.csv",
            "invalid_rows": 8,
            "invalid_percent": 80.0,
            "criticality": "high",
            "error_counts": {"invalid_range": 8},
            "gx_report_path": "data/gx/report.html",
        },
    }

    alert = ingestion_dag.build_alert_message(validation_result)

    assert alert["status"] == "alert_ready"
    assert alert["criticality"] == "high"
    assert "batch_999.csv" in alert["message"]
    assert "data/gx/report.html" in alert["message"]
    assert alert["report_path"] == "data/gx/report.html"


def test_send_alert_skips_when_webhook_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("TEAMS_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    alert = {
        "status": "alert_ready",
        "criticality": "high",
        "message": "High data quality issue",
        "error_counts": {"invalid_range": 8},
    }

    delivery = ingestion_dag.send_alert(alert)

    assert delivery["delivery_status"] == "skipped"
    assert "webhook URL" in delivery["delivery_reason"]


def test_send_alert_posts_to_configured_webhook(monkeypatch) -> None:
    calls = []

    class DummyResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return DummyResponse()

    monkeypatch.setenv("TEAMS_WEBHOOK_URL", "https://example.test/webhook")
    monkeypatch.setattr(ingestion_dag.requests, "post", fake_post)
    alert = {
        "status": "alert_ready",
        "criticality": "high",
        "message": "High data quality issue",
        "error_counts": {"invalid_range": 8},
    }

    delivery = ingestion_dag.send_alert(alert)

    assert delivery["delivery_status"] == "sent"
    assert delivery["status_code"] == 200
    assert calls == [
        {
            "url": "https://example.test/webhook",
            "json": {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Bike sharing data quality alert",
                "themeColor": "E81123",
                "title": "Bike Sharing Data Quality Alert",
                "text": "High data quality issue",
            },
            "timeout": 10,
        }
    ]


def test_save_statistics_can_use_postgres_hook(monkeypatch) -> None:
    calls = []

    class FakePostgresHook:
        def __init__(self, postgres_conn_id):
            calls.append(("conn_id", postgres_conn_id))

        def run(self, sql):
            calls.append(("run", "CREATE TABLE" in sql))

        def get_first(self, sql, parameters):
            calls.append(("insert", parameters))
            return (123, "high")

    monkeypatch.setattr(ingestion_dag, "PostgresHook", FakePostgresHook)
    summary = ingestion_dag.ValidationSummary(
        file_path="batch_999.csv",
        total_rows=10,
        valid_rows=2,
        invalid_rows=8,
        invalid_percent=80.0,
        criticality="high",
        missing_columns=["dteday"],
        error_counts={"invalid_range": 8},
        gx_success=False,
        gx_failed_expectations=2,
        gx_report_path="data/gx/report.html",
    )

    result = ingestion_dag.save_statistics_with_postgres_hook(
        summary,
        postgres_conn_id="dsp_postgres",
    )

    assert result == {
        "status": "saved",
        "stat_id": 123,
        "criticality": "high",
        "database_method": "postgres_hook",
        "postgres_conn_id": "dsp_postgres",
    }
    assert calls[0] == ("conn_id", "dsp_postgres")
    assert calls[1] == ("run", True)
    assert calls[2][0] == "insert"

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.ingestion.pipeline import (
    choose_random_raw_file,
    save_statistics,
    split_and_save_data,
)
from app.validation.bike_data import ValidationSummary, validate_bike_data

try:
    from airflow.sdk import DAG, task
except ImportError:  # pragma: no cover - Airflow is installed in Docker later.
    DAG = None
    task = None

try:
    from airflow.providers.postgres.hooks.postgres import PostgresHook
except ImportError:  # pragma: no cover - Airflow is installed in Docker later.
    PostgresHook = None


DAG_ID = "bike_data_ingestion"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
DEFAULT_GOOD_DATA_DIR = PROJECT_ROOT / "data" / "good_data"
DEFAULT_BAD_DATA_DIR = PROJECT_ROOT / "data" / "bad_data"
DEFAULT_POSTGRES_CONN_ID = "dsp_postgres"
MAX_ACTIVE_RUNS = 1

TASK_IDS = [
    "read_data",
    "validate_data",
    "save_statistics",
    "send_alerts",
    "split_and_save_data",
]


def get_ingestion_config() -> dict[str, str]:
    return {
        "raw_data_dir": os.getenv(
            "INGESTION_RAW_DATA_DIR",
            os.getenv("RAW_DATA_DIR", str(DEFAULT_RAW_DATA_DIR)),
        ),
        "good_data_dir": os.getenv(
            "INGESTION_GOOD_DATA_DIR",
            os.getenv("GOOD_DATA_DIR", str(DEFAULT_GOOD_DATA_DIR)),
        ),
        "bad_data_dir": os.getenv(
            "INGESTION_BAD_DATA_DIR",
            os.getenv("BAD_DATA_DIR", str(DEFAULT_BAD_DATA_DIR)),
        ),
    }


def read_random_raw_file(raw_data_dir: str) -> dict[str, Any]:
    selected_file = choose_random_raw_file(Path(raw_data_dir))
    if selected_file is None:
        return {"status": "skipped", "reason": "No raw data files found"}

    csv_content = selected_file.read_text(encoding="utf-8")
    payload = {
        "status": "read",
        "file_path": str(selected_file),
        "file_name": selected_file.name,
        "csv_content": csv_content,
    }
    selected_file.unlink()
    return payload


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def validate_file(read_result: dict[str, Any]) -> dict[str, Any]:
    if read_result.get("status") != "read":
        return {
            "status": "skipped",
            "reason": read_result.get("reason", "No raw data was read"),
        }

    file_name = str(read_result["file_name"])
    processing_dir = PROJECT_ROOT / "data" / "processing_data"
    processing_dir.mkdir(parents=True, exist_ok=True)
    processing_file = processing_dir / file_name
    processing_file.write_text(read_result["csv_content"], encoding="utf-8")
    try:
        summary, good_df, bad_df = validate_bike_data(processing_file)
    finally:
        processing_file.unlink(missing_ok=True)
    summary.file_path = str(read_result["file_path"])
    return {
        "status": "validated",
        "summary": summary.to_dict(),
        "columns": list(good_df.columns),
        "good_records": dataframe_records(good_df),
        "bad_records": dataframe_records(bad_df),
    }


def save_ingestion_statistics(validation_result: dict[str, Any]) -> dict:
    if validation_result.get("status") != "validated":
        return {
            "status": "skipped",
            "reason": validation_result.get("reason", "No validation result"),
        }

    summary = ValidationSummary(**validation_result["summary"])
    postgres_conn_id = os.getenv("AIRFLOW_POSTGRES_CONN_ID")
    if PostgresHook is not None and postgres_conn_id:
        return save_statistics_with_postgres_hook(summary, postgres_conn_id)

    from app.database.models import Base
    from app.database.session import SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        stat = save_statistics(db, summary)
        return {
            "status": "saved",
            "stat_id": stat.id,
            "criticality": stat.criticality,
            "database_method": "sqlalchemy",
        }


def save_statistics_with_postgres_hook(
    summary: ValidationSummary,
    postgres_conn_id: str = DEFAULT_POSTGRES_CONN_ID,
) -> dict:
    if PostgresHook is None:
        raise RuntimeError("PostgresHook is not available")

    hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    hook.run(
        """
        CREATE TABLE IF NOT EXISTS ingestion_stats (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            total_rows INTEGER NOT NULL,
            valid_rows INTEGER NOT NULL,
            invalid_rows INTEGER NOT NULL,
            invalid_percent DOUBLE PRECISION NOT NULL,
            criticality VARCHAR(32) NOT NULL,
            missing_columns JSONB NOT NULL,
            error_counts JSONB NOT NULL,
            gx_success BOOLEAN NOT NULL,
            gx_failed_expectations INTEGER NOT NULL,
            gx_report_path VARCHAR(1024),
            created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
        );
        """
    )
    row = hook.get_first(
        """
        INSERT INTO ingestion_stats (
            file_name,
            total_rows,
            valid_rows,
            invalid_rows,
            invalid_percent,
            criticality,
            missing_columns,
            error_counts,
            gx_success,
            gx_failed_expectations,
            gx_report_path
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
        RETURNING id, criticality;
        """,
        parameters=(
            Path(summary.file_path).name,
            summary.total_rows,
            summary.valid_rows,
            summary.invalid_rows,
            summary.invalid_percent,
            summary.criticality,
            json.dumps(summary.missing_columns),
            json.dumps(summary.error_counts),
            summary.gx_success,
            summary.gx_failed_expectations,
            summary.gx_report_path,
        ),
    )
    return {
        "status": "saved",
        "stat_id": row[0],
        "criticality": row[1],
        "database_method": "postgres_hook",
        "postgres_conn_id": postgres_conn_id,
    }


def build_alert_message(validation_result: dict[str, Any]) -> dict:
    if validation_result.get("status") != "validated":
        return {
            "status": "skipped",
            "reason": validation_result.get("reason", "No validation result"),
        }

    summary = validation_result["summary"]
    criticality = summary["criticality"]
    if criticality in {"none", "low"}:
        return {
            "status": "skipped",
            "reason": "No medium or high criticality issue detected",
            "criticality": criticality,
        }

    return {
        "status": "alert_ready",
        "criticality": criticality,
        "message": (
            f"{criticality.upper()} data quality issue in "
            f"{Path(summary['file_path']).name}: "
            f"{summary['invalid_rows']} invalid rows "
            f"({summary['invalid_percent']}%). "
            f"Report: {summary.get('gx_report_path') or 'not available'}."
        ),
        "error_counts": summary["error_counts"],
        "report_path": summary.get("gx_report_path"),
    }


def send_alert(alert: dict[str, Any]) -> dict[str, Any]:
    if alert.get("status") != "alert_ready":
        return alert

    webhook_url = os.getenv("TEAMS_WEBHOOK_URL") or os.getenv(
        "ALERT_WEBHOOK_URL"
    )
    if not webhook_url:
        return {
            **alert,
            "delivery_status": "skipped",
            "delivery_reason": "No alert webhook URL configured",
        }

    response = requests.post(
        webhook_url,
        json={
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": "Bike sharing data quality alert",
            "themeColor": "E81123",
            "title": "Bike Sharing Data Quality Alert",
            "text": alert["message"],
        },
        timeout=10,
    )
    response.raise_for_status()
    return {
        **alert,
        "delivery_status": "sent",
        "status_code": response.status_code,
    }


def split_valid_and_invalid_data(
    validation_result: dict[str, Any],
    good_data_dir: str,
    bad_data_dir: str,
) -> dict[str, str | None]:
    if validation_result.get("status") != "validated":
        return {"good_file": None, "bad_file": None}

    summary = ValidationSummary(**validation_result["summary"])
    columns = validation_result["columns"]
    good_df = pd.DataFrame(
        validation_result["good_records"],
        columns=columns,
    )
    bad_df = pd.DataFrame(
        validation_result["bad_records"],
        columns=columns,
    )
    return split_and_save_data(
        source_file=Path(summary.file_path),
        summary=summary,
        good_df=good_df,
        bad_df=bad_df,
        good_data_dir=Path(good_data_dir),
        bad_data_dir=Path(bad_data_dir),
    )


if DAG is not None and task is not None:
    with DAG(
        dag_id=DAG_ID,
        description="Validate incoming bike-sharing data and save stats.",
        schedule="*/1 * * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=MAX_ACTIVE_RUNS,
        tags=["bike-sharing", "ingestion", "data-quality"],
    ) as ingestion_dag:

        @task(task_id="read_data")
        def read_data_task() -> dict[str, Any]:
            config = get_ingestion_config()
            return {
                **config,
                **read_random_raw_file(config["raw_data_dir"]),
            }

        @task(task_id="validate_data")
        def validate_data_task(payload: dict[str, Any]) -> dict[str, Any]:
            return validate_file(payload)

        @task(task_id="save_statistics")
        def save_statistics_task(validation_result: dict[str, Any]) -> dict:
            return save_ingestion_statistics(validation_result)

        @task(task_id="send_alerts")
        def send_alerts_task(validation_result: dict[str, Any]) -> dict:
            alert = build_alert_message(validation_result)
            delivery = send_alert(alert)
            print(json.dumps(delivery, indent=2))
            return delivery

        @task(task_id="split_and_save_data", multiple_outputs=False)
        def split_and_save_data_task(
            validation_result: dict[str, Any],
            config: dict[str, Any],
        ) -> dict[str, str | None]:
            return split_valid_and_invalid_data(
                validation_result=validation_result,
                good_data_dir=config["good_data_dir"],
                bad_data_dir=config["bad_data_dir"],
            )

        selected_payload = read_data_task()
        validation_payload = validate_data_task(selected_payload)
        [
            save_statistics_task(validation_payload),
            send_alerts_task(validation_payload),
            split_and_save_data_task(validation_payload, selected_payload),
        ]
else:
    ingestion_dag = None

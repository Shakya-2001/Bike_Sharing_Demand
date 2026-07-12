from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from app.database.models import Base
from app.database.session import SessionLocal, engine
from app.ingestion.pipeline import (
    choose_random_raw_file,
    save_statistics,
    split_and_save_data,
)
from app.validation.bike_data import ValidationSummary, validate_bike_data
from scripts.split_dataset import split_dataset

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
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
DEFAULT_GOOD_DATA_DIR = PROJECT_ROOT / "data" / "good_data"
DEFAULT_BAD_DATA_DIR = PROJECT_ROOT / "data" / "bad_data"
DEFAULT_NUMBER_OF_FILES = 25
DEFAULT_POSTGRES_CONN_ID = "dsp_postgres"

TASK_IDS = [
    "split_dataset",
    "read_data",
    "validate_data",
    "save_statistics",
    "send_alerts",
    "split_and_save_data",
]


def get_ingestion_config() -> dict[str, str | int]:
    return {
        "dataset_path": os.getenv(
            "INGESTION_DATASET_PATH",
            str(DEFAULT_DATASET_PATH),
        ),
        "raw_data_dir": os.getenv(
            "INGESTION_RAW_DATA_DIR",
            str(DEFAULT_RAW_DATA_DIR),
        ),
        "good_data_dir": os.getenv(
            "INGESTION_GOOD_DATA_DIR",
            str(DEFAULT_GOOD_DATA_DIR),
        ),
        "bad_data_dir": os.getenv(
            "INGESTION_BAD_DATA_DIR",
            str(DEFAULT_BAD_DATA_DIR),
        ),
        "number_of_files": int(
            os.getenv("INGESTION_NUMBER_OF_FILES", DEFAULT_NUMBER_OF_FILES)
        ),
    }


def create_raw_data_files(config: dict[str, str | int]) -> list[str]:
    raw_data_dir = Path(str(config["raw_data_dir"]))
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(raw_data_dir.glob("*.csv"))
    if existing_files:
        return [str(path) for path in existing_files]

    dataset_path = Path(str(config["dataset_path"]))
    if not dataset_path.exists():
        return []

    generated_files = split_dataset(
        dataset_path=dataset_path,
        output_dir=raw_data_dir,
        number_of_files=int(config["number_of_files"]),
    )
    return [str(path) for path in generated_files]


def read_random_raw_file(raw_data_dir: str) -> str | None:
    selected_file = choose_random_raw_file(Path(raw_data_dir))
    return str(selected_file) if selected_file is not None else None


def validate_file(file_path: str | None) -> dict[str, Any]:
    if file_path is None:
        return {
            "status": "skipped",
            "reason": "No raw data files found",
        }

    summary, _, _ = validate_bike_data(Path(file_path))
    return {
        "status": "validated",
        "summary": summary.to_dict(),
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
    _, good_df, bad_df = validate_bike_data(Path(summary.file_path))
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
        tags=["bike-sharing", "ingestion", "data-quality"],
    ) as ingestion_dag:

        @task(task_id="split_dataset")
        def split_dataset_task() -> dict[str, str | int | list[str]]:
            config = get_ingestion_config()
            files = create_raw_data_files(config)
            return {**config, "raw_files": files}

        @task(task_id="read_data")
        def read_data_task(config: dict[str, Any]) -> dict[str, Any]:
            selected_file = read_random_raw_file(config["raw_data_dir"])
            return {**config, "selected_file": selected_file}

        @task(task_id="validate_data")
        def validate_data_task(payload: dict[str, Any]) -> dict[str, Any]:
            return validate_file(payload["selected_file"])

        @task(task_id="save_statistics")
        def save_statistics_task(validation_result: dict[str, Any]) -> dict:
            return save_ingestion_statistics(validation_result)

        @task(task_id="send_alerts")
        def send_alerts_task(validation_result: dict[str, Any]) -> dict:
            alert = build_alert_message(validation_result)
            delivery = send_alert(alert)
            print(json.dumps(delivery, indent=2))
            return delivery

        @task(task_id="split_and_save_data")
        def split_and_save_data_task(
            validation_result: dict[str, Any],
            config: dict[str, Any],
        ) -> dict[str, str | None]:
            return split_valid_and_invalid_data(
                validation_result=validation_result,
                good_data_dir=config["good_data_dir"],
                bad_data_dir=config["bad_data_dir"],
            )

        ingestion_config = split_dataset_task()
        selected_payload = read_data_task(ingestion_config)
        validation_payload = validate_data_task(selected_payload)
        [
            save_statistics_task(validation_payload),
            send_alerts_task(validation_payload),
            split_and_save_data_task(validation_payload, ingestion_config),
        ]
else:
    ingestion_dag = None

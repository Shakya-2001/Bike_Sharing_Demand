from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from airflow.exceptions import AirflowSkipException
    from airflow.sdk import DAG, task
except ImportError:  # pragma: no cover - Airflow is installed in Docker.
    DAG = None
    task = None

    class AirflowSkipException(Exception):
        pass


DAG_ID = "bike_demand_prediction"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOOD_DATA_DIR = PROJECT_ROOT / "data" / "good_data"
DEFAULT_STATE_FILE = (
    PROJECT_ROOT / "data" / "prediction_state" / "processed_files.json"
)
DEFAULT_API_URL = "http://api:8000"
MAX_ACTIVE_RUNS = 1
TASK_IDS = ["check_for_new_data", "make_predictions"]

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


def get_prediction_config() -> dict[str, str]:
    return {
        "good_data_dir": os.getenv(
            "PREDICTION_GOOD_DATA_DIR",
            os.getenv("GOOD_DATA_DIR", str(DEFAULT_GOOD_DATA_DIR)),
        ),
        "state_file": os.getenv(
            "PREDICTION_STATE_FILE",
            str(DEFAULT_STATE_FILE),
        ),
        "api_url": os.getenv("PREDICTION_API_URL", DEFAULT_API_URL),
    }


def read_processed_files(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Cannot read prediction state: {exc}") from exc
    return set(payload.get("processed_files", []))


def find_new_good_files(
    good_data_dir: str,
    state_file: str,
) -> list[str]:
    data_dir = Path(good_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    processed = read_processed_files(Path(state_file))
    return [
        str(path)
        for path in sorted(data_dir.glob("*.csv"))
        if str(path.resolve()) not in processed
    ]


def check_for_new_data(config: dict[str, str]) -> dict[str, Any]:
    files = find_new_good_files(
        config["good_data_dir"],
        config["state_file"],
    )
    if not files:
        raise AirflowSkipException("No newly ingested good-data files found")
    return {**config, "files": files}


def load_batch_rows(file_paths: list[str]) -> list[dict[str, Any]]:
    frames = []
    for file_path in file_paths:
        frame = pd.read_csv(file_path)
        missing = [column for column in FEATURE_COLUMNS if column not in frame]
        if missing:
            raise ValueError(
                f"{Path(file_path).name} is missing columns: "
                f"{', '.join(missing)}"
            )
        frames.append(frame[FEATURE_COLUMNS])

    if not frames:
        raise ValueError("No files were provided for prediction")

    batch = pd.concat(frames, ignore_index=True)
    if batch.empty:
        raise ValueError("New good-data files contain no rows")
    if batch.isnull().any().any():
        raise ValueError("Prediction features cannot contain null values")
    return batch.to_dict(orient="records")


def write_processed_files(state_file: Path, file_paths: list[str]) -> None:
    processed = read_processed_files(state_file)
    processed.update(str(Path(path).resolve()) for path in file_paths)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = state_file.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps({"processed_files": sorted(processed)}, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(state_file)


def make_batch_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload["files"]
    rows = load_batch_rows(files)
    endpoint = f"{payload['api_url'].rstrip('/')}/predict"
    response = requests.post(
        endpoint,
        json={"features": rows, "source": "scheduled"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()

    predictions = result.get("predictions", [])
    if len(predictions) != len(rows):
        raise RuntimeError(
            "Prediction API returned a different number of predictions"
        )

    write_processed_files(Path(payload["state_file"]), files)
    return {
        "status": "predicted",
        "files_processed": len(files),
        "rows_predicted": len(predictions),
        "model_name": result.get("model_name"),
    }


if DAG is not None and task is not None:
    with DAG(
        dag_id=DAG_ID,
        description="Make scheduled predictions from new good-data files.",
        schedule="*/2 * * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=MAX_ACTIVE_RUNS,
        tags=["bike-sharing", "prediction"],
    ) as prediction_dag:

        @task(task_id="check_for_new_data")
        def check_for_new_data_task() -> dict[str, Any]:
            return check_for_new_data(get_prediction_config())

        @task(task_id="make_predictions")
        def make_predictions_task(payload: dict[str, Any]) -> dict[str, Any]:
            result = make_batch_predictions(payload)
            print(json.dumps(result, indent=2))
            return result

        make_predictions_task(check_for_new_data_task())
else:
    prediction_dag = None

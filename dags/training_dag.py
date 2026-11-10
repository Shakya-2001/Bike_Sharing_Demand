from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.training.config import get_training_config

try:
    from airflow.exceptions import AirflowSkipException
    from airflow.sdk import DAG, task
except ImportError:  # pragma: no cover - Airflow is installed in Docker.
    DAG = None
    task = None

    class AirflowSkipException(Exception):
        pass


DAG_ID = "bike_model_training"
MAX_ACTIVE_RUNS = 1
TASK_IDS = [
    "load_data",
    "train_model",
    "save_training_stats",
    "evaluate_candidate",
    "promote_to_champion",
    "notify_api_reload",
    "archive_data",
]

# Heavy libraries (pandas, scikit-learn, mlflow) are imported inside these
# functions, not at the top of the file, so Airflow parses the DAG quickly
# and each task process uses less memory.


def load_training_data() -> dict[str, Any]:
    from app.training import pipeline, registry

    has_champion = registry.get_champion_version() is not None
    result = pipeline.check_for_training_data(
        get_training_config(),
        has_champion,
    )
    if not result["should_train"]:
        raise AirflowSkipException(result["reason"])
    return result


def run_train_model(payload: dict[str, Any]) -> dict[str, Any]:
    from app.training import pipeline

    return pipeline.train_candidate(
        get_training_config(),
        payload["good_files"],
    )


def run_save_stats(train_result: dict[str, Any]) -> dict[str, Any]:
    from app.training import pipeline

    return pipeline.save_candidate_stats(train_result)


def run_evaluate(train_result: dict[str, Any]) -> dict[str, Any]:
    from app.training import pipeline

    return pipeline.evaluate_candidate(get_training_config(), train_result)


def run_promote(decision: dict[str, Any]) -> dict[str, Any]:
    from app.training import pipeline

    if not decision["promote"]:
        raise AirflowSkipException(
            "Candidate rejected: " + "; ".join(decision["reasons"])
        )
    return pipeline.promote_candidate(decision)


def run_notify_reload(promotion: dict[str, Any]) -> dict[str, Any]:
    from app.training import pipeline

    return pipeline.reload_api(get_training_config()["api_url"])


def run_archive(promotion: dict[str, Any]) -> list[str]:
    from app.training import pipeline

    config = get_training_config()
    return pipeline.archive_files(
        Path(config["good_data_dir"]),
        Path(config["archived_data_dir"]),
        promotion["good_files"],
    )


if DAG is not None and task is not None:
    with DAG(
        dag_id=DAG_ID,
        description="Retrain the bike demand model and promote it if better.",
        schedule="*/10 * * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=MAX_ACTIVE_RUNS,
        tags=["bike-sharing", "training", "mlflow"],
    ) as training_dag:

        # multiple_outputs=False keeps each result as ONE XCom. Airflow
        # would otherwise store every key of a returned dict separately
        # and cannot store keys whose value is None.
        @task(task_id="load_data", multiple_outputs=False)
        def load_data_task() -> dict[str, Any]:
            return load_training_data()

        @task(task_id="train_model", multiple_outputs=False)
        def train_model_task(payload: dict[str, Any]) -> dict[str, Any]:
            return run_train_model(payload)

        @task(task_id="save_training_stats", multiple_outputs=False)
        def save_stats_task(train_result: dict[str, Any]) -> dict[str, Any]:
            return run_save_stats(train_result)

        @task(task_id="evaluate_candidate", multiple_outputs=False)
        def evaluate_task(train_result: dict[str, Any]) -> dict[str, Any]:
            decision = run_evaluate(train_result)
            print(json.dumps(decision, indent=2))
            return decision

        @task(task_id="promote_to_champion", multiple_outputs=False)
        def promote_task(decision: dict[str, Any]) -> dict[str, Any]:
            return run_promote(decision)

        @task(task_id="notify_api_reload", multiple_outputs=False)
        def notify_reload_task(promotion: dict[str, Any]) -> dict[str, Any]:
            return run_notify_reload(promotion)

        @task(task_id="archive_data", multiple_outputs=False)
        def archive_task(promotion: dict[str, Any]) -> list[str]:
            return run_archive(promotion)

        payload = load_data_task()
        train_result = train_model_task(payload)
        stats = save_stats_task(train_result)
        decision = evaluate_task(train_result)
        stats >> decision
        promotion = promote_task(decision)
        [notify_reload_task(promotion), archive_task(promotion)]
else:
    training_dag = None

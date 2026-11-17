from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow.exceptions import AirflowSkipException
from airflow.sdk import DAG, get_current_context, task


with DAG(
    dag_id="bike_demand_training",
    description="Train, evaluate, and conditionally promote bike models.",
    schedule=os.getenv("TRAINING_SCHEDULE", "*/10 * * * *"),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["bike-sharing", "training", "mlflow"],
) as training_dag:

    @task(task_id="load_data")
    def load_data_task() -> dict:
        from app.training.snapshots import create_training_snapshot

        run_id = get_current_context()["run_id"]
        snapshot = create_training_snapshot(run_id)

        if snapshot["status"] != "ready":
            raise AirflowSkipException(
                f"Only {snapshot['new_rows']} new rows available. "
                f"Required: {snapshot['required_rows']}."
            )
        return snapshot

    @task(task_id="train_model", retries=0)
    def train_model_task(snapshot: dict) -> dict:
        from app.training.pipeline import train_candidate
        from app.training.snapshots import load_training_snapshot

        prepared = load_training_snapshot(snapshot)
        return train_candidate(prepared)

    @task(task_id="save_training_stats")
    def save_training_stats_task(
        candidate: dict,
        snapshot: dict,
    ) -> dict:
        from app.training.persistence import save_training_statistics
        from app.training.snapshots import record_training_attempt

        saved = save_training_statistics(candidate)
        record_training_attempt(snapshot, saved)
        return saved

    @task(task_id="evaluate_candidate")
    def evaluate_candidate_task(candidate: dict) -> dict:
        from app.training.evaluation import evaluate_candidate

        return evaluate_candidate(candidate)

    @task.branch(task_id="choose_promotion")
    def choose_promotion_task(candidate: dict) -> str:
        if candidate["evaluation"]["eligible"]:
            return "promote_to_champion"
        return "notify_rejection"

    @task(task_id="promote_to_champion")
    def promote_to_champion_task(candidate: dict) -> dict:
        from app.training.promotion import promote_to_champion

        return promote_to_champion(candidate)

    @task(task_id="notify_rejection")
    def notify_rejection_task(candidate: dict) -> dict:
        from app.training.alerts import send_rejection_alert

        return send_rejection_alert(candidate)

    @task(task_id="notify_api_reload")
    def notify_api_reload_task(candidate: dict) -> dict:
        from app.training.deployment import notify_api_reload

        return notify_api_reload(candidate)

    @task(task_id="archive_data", retries=5)
    def archive_data_task(candidate: dict) -> dict:
        from app.training.archiving import archive_training_files

        return archive_training_files(candidate)

    snapshot = load_data_task()
    trained = train_model_task(snapshot)
    saved = save_training_stats_task(trained, snapshot)
    evaluated = evaluate_candidate_task(saved)

    decision = choose_promotion_task(evaluated)
    promoted = promote_to_champion_task(evaluated)
    rejected = notify_rejection_task(evaluated)

    decision >> [promoted, rejected]

    notify_api_reload_task(promoted)
    archive_data_task(promoted)

from __future__ import annotations

import os
from contextlib import closing

from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


def promote_to_champion(candidate: dict) -> dict:
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    evaluation = candidate["evaluation"]
    if not evaluation["eligible"]:
        raise ValueError("Candidate did not pass promotion criteria")

    model_name = candidate["model_name"]
    version = candidate["model_version"]
    expected_champion = evaluation["champion_version"]

    client = MlflowClient(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
    )
    registered = client.get_model_version(model_name, version)
    if registered.run_id != candidate["mlflow_run_id"]:
        raise ValueError("Candidate version and training run do not match")
    if registered.status != "READY":
        raise ValueError("Candidate model version is not ready")

    hook = PostgresHook(
        postgres_conn_id=os.getenv(
            "AIRFLOW_POSTGRES_CONN_ID",
            "dsp_postgres",
        )
    )

    with closing(hook.get_conn()) as connection:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (model_name,),
            )
            cursor.execute(
                """
                SELECT mlflow_run_id
                FROM training_stats
                WHERE model_name = %s AND model_version = %s
                FOR UPDATE
                """,
                (model_name, version),
            )
            record = cursor.fetchone()
            if record is None or record[0] != candidate["mlflow_run_id"]:
                raise ValueError("Matching training statistics are required")

            try:
                champion = client.get_model_version_by_alias(
                    model_name,
                    "champion",
                )
                current_version = str(champion.version)
            except RestException as exc:
                if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
                current_version = None

            if current_version not in (expected_champion, version):
                raise RuntimeError(
                    "Champion changed after evaluation. Evaluate again."
                )

            if current_version != version:
                client.set_registered_model_alias(
                    model_name,
                    "champion",
                    version,
                )

            cursor.execute(
                """
                UPDATE training_stats
                SET promoted_at = COALESCE(promoted_at, NOW())
                WHERE model_name = %s AND model_version = %s
                """,
                (model_name, version),
            )

    client.set_model_version_tag(
        model_name,
        version,
        "promotion_status",
        "promoted",
    )
    return {**candidate, "promotion_status": "promoted"}

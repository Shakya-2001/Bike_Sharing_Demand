from __future__ import annotations

import json
import os


def save_training_statistics(candidate: dict) -> dict:
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    hook = PostgresHook(
        postgres_conn_id=os.getenv(
            "AIRFLOW_POSTGRES_CONN_ID",
            "dsp_postgres",
        )
    )
    hook.run(
        """
        INSERT INTO training_stats (
            model_name,
            model_version,
            mlflow_run_id,
            training_rows,
            numerical_stats,
            categorical_stats
        )
        VALUES (%s, %s, %s, %s, %s::json, %s::json)
        ON CONFLICT (model_name, model_version) DO NOTHING
        """,
        parameters=(
            candidate["model_name"],
            candidate["model_version"],
            candidate["mlflow_run_id"],
            candidate["training_rows"],
            json.dumps(candidate["numerical_stats"]),
            json.dumps(candidate["categorical_stats"]),
        ),
    )
    return candidate

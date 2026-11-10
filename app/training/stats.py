"""Feature statistics of the training data (the drift baseline).

One row per numeric feature, or per category of a categorical feature, for
each trained model version. `is_baseline` is true only for the promoted
(champion) version: Grafana compares live data with those rows.

The database is reached with Airflow's PostgresHook (like the ingestion
job): the Airflow image ships SQLAlchemy 1.4, so the SQLAlchemy 2 models of
the API cannot be imported there.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from app.training.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

TABLE_NAME = "training_stats"
DEFAULT_CONNECTION_ID = "dsp_postgres"

CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS training_stats (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(128) NOT NULL,
        model_version INTEGER NOT NULL,
        run_id VARCHAR(64) NOT NULL,
        feature_name VARCHAR(64) NOT NULL,
        feature_type VARCHAR(16) NOT NULL,
        category VARCHAR(64),
        mean DOUBLE PRECISION,
        std DOUBLE PRECISION,
        min_value DOUBLE PRECISION,
        max_value DOUBLE PRECISION,
        proportion DOUBLE PRECISION,
        row_count INTEGER NOT NULL,
        is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_training_stats_version "
    "ON training_stats (model_version)",
    "CREATE INDEX IF NOT EXISTS ix_training_stats_baseline "
    "ON training_stats (is_baseline)",
]

INSERT_FIELDS = [
    "model_name",
    "model_version",
    "run_id",
    "feature_name",
    "feature_type",
    "category",
    "mean",
    "std",
    "min_value",
    "max_value",
    "proportion",
    "row_count",
    "is_baseline",
]


def get_hook() -> Any:
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    return PostgresHook(
        postgres_conn_id=os.getenv(
            "AIRFLOW_POSTGRES_CONN_ID",
            DEFAULT_CONNECTION_ID,
        )
    )


def compute_training_stats(train_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Mean/std/min/max of numeric columns and category shares.

    The target `cnt` is included as a numeric column so predictions can be
    compared with what the model saw in training.
    """
    stats: list[dict[str, Any]] = []
    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        series = train_df[column].astype(float)
        stats.append(
            {
                "feature_name": column,
                "feature_type": "numeric",
                "category": None,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "proportion": None,
            }
        )

    total = len(train_df)
    for column in CATEGORICAL_COLUMNS:
        shares = train_df[column].value_counts()
        for category, count in shares.items():
            stats.append(
                {
                    "feature_name": column,
                    "feature_type": "categorical",
                    "category": str(category),
                    "mean": None,
                    "std": None,
                    "min_value": None,
                    "max_value": None,
                    "proportion": float(count / total),
                }
            )
    return stats


def ensure_table(hook: Any) -> None:
    hook.run(CREATE_STATEMENTS)


def save_training_stats(
    hook: Any,
    stats: list[dict[str, Any]],
    model_name: str,
    model_version: int,
    run_id: str,
    row_count: int,
) -> int:
    """Store the stats of one candidate (never as baseline yet)."""
    ensure_table(hook)
    hook.run(
        "DELETE FROM training_stats WHERE run_id = %s",
        parameters=(run_id,),
    )
    rows = [
        (
            model_name,
            model_version,
            run_id,
            item["feature_name"],
            item["feature_type"],
            item["category"],
            item["mean"],
            item["std"],
            item["min_value"],
            item["max_value"],
            item["proportion"],
            row_count,
            False,
        )
        for item in stats
    ]
    hook.insert_rows(
        table=TABLE_NAME,
        rows=rows,
        target_fields=INSERT_FIELDS,
        commit_every=1000,
    )
    return len(rows)


def mark_baseline(hook: Any, model_name: str, model_version: int) -> int:
    """Make the promoted version the only drift baseline."""
    ensure_table(hook)
    hook.run(
        "UPDATE training_stats SET is_baseline = FALSE "
        "WHERE model_name = %s",
        parameters=(model_name,),
    )
    hook.run(
        "UPDATE training_stats SET is_baseline = TRUE "
        "WHERE model_name = %s AND model_version = %s",
        parameters=(model_name, model_version),
    )
    row = hook.get_first(
        "SELECT COUNT(*) FROM training_stats "
        "WHERE model_name = %s AND is_baseline",
        parameters=(model_name,),
    )
    return int(row[0])

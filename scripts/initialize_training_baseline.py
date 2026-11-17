from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy.dialects.postgresql import insert

from app.database.models import TrainingStat
from app.database.session import SessionLocal
from app.training.statistics import calculate_feature_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_name = os.environ["MLFLOW_MODEL_NAME"]
    client = MlflowClient(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
    )
    champion = client.get_model_version_by_alias(
        model_name,
        "champion",
    )
    if not champion.run_id:
        raise ValueError("Champion has no associated MLflow run")

    run = client.get_run(champion.run_id)
    if run.data.tags.get("mlflow.runName") != (
        "register-defense1-baseline"
    ):
        raise ValueError(
            "This script only initializes the original Defense 1 baseline"
        )

    metadata = json.loads(
        (PROJECT_ROOT / "models" / "baseline_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    data = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "hour.csv")
    data = data.sort_values("instant").reset_index(drop=True)

    if len(data) != metadata["rows"]:
        raise ValueError("Dataset size differs from the baseline metadata")
    if metadata["split"] != "chronological":
        raise ValueError("Expected a chronological baseline split")

    training_data = data.iloc[:int(len(data) * 0.8)]
    numerical, categorical = calculate_feature_statistics(training_data)

    statement = insert(TrainingStat).values(
        model_name=model_name,
        model_version=str(champion.version),
        mlflow_run_id=champion.run_id,
        training_rows=len(training_data),
        numerical_stats=numerical,
        categorical_stats=categorical,
        promoted_at=datetime.now(timezone.utc),
    )
    statement = statement.on_conflict_do_nothing(
        constraint="uq_training_stats_model_version"
    )

    with SessionLocal.begin() as session:
        result = session.execute(statement)

    if result.rowcount:
        print(f"Saved champion baseline for version {champion.version}")
    else:
        print("Baseline already exists. Existing statistics retained.")

    print(f"Original champion training rows: {len(training_data)}")


if __name__ == "__main__":
    main()

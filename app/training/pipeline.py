from __future__ import annotations

import os
from time import perf_counter

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from app.training.statistics import calculate_feature_statistics
from scripts.train_baseline import FEATURE_COLUMNS, build_model


def train_candidate(prepared_data: dict) -> dict:
    if prepared_data["status"] != "ready":
        raise ValueError("Training requires enough new labeled data")

    model_name = os.environ["MLFLOW_MODEL_NAME"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("bike-demand-training")

    training = prepared_data["training"]
    features = training[FEATURE_COLUMNS]
    target = training["cnt"]
    numerical, categorical = calculate_feature_statistics(training)

    with mlflow.start_run(run_name="train-bike-demand-candidate") as run:
        mlflow.log_params({
            "training_rows": len(training),
            "new_labeled_rows": prepared_data["new_rows"],
            "test_rows": len(prepared_data["test"]),
            "test_sha256": prepared_data["test_sha256"],
            "primary_metric": "rmse",
            "model_type": "HistGradientBoostingRegressor",
        })
        mlflow.log_dict(
            {"files": prepared_data["files"]},
            "training_files.json",
        )

        model = build_model()
        started = perf_counter()
        model.fit(features, target)
        elapsed = perf_counter() - started
        mlflow.log_metric("training_duration_seconds", elapsed)

        sample = features.head(5)
        signature = infer_signature(sample, model.predict(sample))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=sample,
        )
        version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name,
        )

        result = {
            "model_name": model_name,
            "model_version": str(version.version),
            "mlflow_run_id": run.info.run_id,
            "training_rows": len(training),
            "new_rows": prepared_data["new_rows"],
            "test_sha256": prepared_data["test_sha256"],
            "numerical_stats": numerical,
            "categorical_stats": categorical,
            "files": prepared_data["files"],
        }
        mlflow.log_dict(result, "candidate_summary.json")

    return result

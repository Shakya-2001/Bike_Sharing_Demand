"""MLflow tracking and model registry calls.

mlflow is imported inside the functions so the rest of the code (and the
unit tests) work without it, and Airflow parses the DAG file faster.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from app.training.config import CHAMPION_ALIAS, EXPERIMENT_NAME, MODEL_NAME


def configure_mlflow():
    import mlflow
    import mlflow.sklearn  # noqa: F401

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def is_missing_champion(error_code: str, message: str) -> bool:
    """True if MLflow says the champion alias (or model) does not exist.

    MLflow 3.1 answers INVALID_PARAMETER_VALUE with "alias ... not found"
    when the alias is missing, so the message is checked too.
    """
    if error_code == "RESOURCE_DOES_NOT_EXIST":
        return True
    text = message.lower()
    return "alias" in text and "not found" in text


def get_champion_version():
    """Return the model version that holds @champion, or None."""
    from mlflow.exceptions import MlflowException

    mlflow = configure_mlflow()
    client = mlflow.MlflowClient()
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as exc:
        if is_missing_champion(str(exc.error_code), str(exc)):
            return None
        raise


def load_model_version(version: str) -> Any:
    mlflow = configure_mlflow()
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def load_champion_model() -> Any:
    mlflow = configure_mlflow()
    return mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )


def log_and_register_candidate(
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    input_example: pd.DataFrame,
) -> dict[str, str]:
    """Log a training run, save the model and register a new version."""
    mlflow = configure_mlflow()
    with mlflow.start_run(run_name="candidate_training") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
        )
        version = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    return {
        "run_id": run.info.run_id,
        "model_version": str(version.version),
    }


def log_comparison(
    run_id: str,
    promote: bool,
    reasons: list[str],
    champion_version: str | None,
    champion_metrics: dict[str, float] | None,
    inference_ms: float,
    threshold_ms: float,
) -> None:
    """Add the champion comparison to the candidate's MLflow run."""
    mlflow = configure_mlflow()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("candidate_inference_ms", inference_ms)
        mlflow.log_param("inference_threshold_ms", threshold_ms)
        if champion_metrics is not None:
            mlflow.log_metric("champion_test_rmse", champion_metrics["rmse"])
            mlflow.log_metric("champion_test_mae", champion_metrics["mae"])
            mlflow.set_tag("champion_version", str(champion_version))
        mlflow.set_tag(
            "promotion_decision",
            "promoted" if promote else "rejected",
        )
        mlflow.set_tag("promotion_reasons", "; ".join(reasons))


def promote_version(version: str) -> None:
    """Move the @champion alias to this model version."""
    mlflow = configure_mlflow()
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

from __future__ import annotations

import os
from time import perf_counter

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

from app.training.data import DATA_DIR, read_labeled_csv
from scripts.train_baseline import FEATURE_COLUMNS


def score_model(model, features, target) -> tuple[float, float]:
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.shape != (len(features),):
        raise ValueError("Model returned an unexpected prediction shape")
    if not np.isfinite(predictions).all():
        raise ValueError("Model returned non-finite predictions")

    rmse = float(mean_squared_error(target, predictions) ** 0.5)

    sample = features.head(100)
    model.predict(sample)
    started = perf_counter()
    for _ in range(3):
        model.predict(sample)
    milliseconds_per_row = (
        (perf_counter() - started) * 1000 / (3 * len(sample))
    )
    return rmse, milliseconds_per_row


def evaluate_candidate(candidate: dict) -> dict:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    threshold = float(
        os.getenv("TRAINING_MAX_INFERENCE_MS_PER_ROW", "10")
    )
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Inference threshold must be positive and finite")

    test, digest = read_labeled_csv(
        DATA_DIR / "training" / "test.csv"
    )
    if digest != candidate["test_sha256"]:
        raise ValueError("Test data changed after candidate training")

    features = test[FEATURE_COLUMNS]
    target = test["cnt"]
    model_name = candidate["model_name"]

    model = mlflow.sklearn.load_model(
        f"models:/{model_name}/{candidate['model_version']}"
    )
    candidate_rmse, inference_ms = score_model(
        model,
        features,
        target,
    )

    try:
        champion = client.get_model_version_by_alias(
            model_name,
            "champion",
        )
    except RestException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        champion = None

    champion_version = None
    champion_rmse = None
    if champion is not None:
        champion_version = str(champion.version)
        champion_model = mlflow.sklearn.load_model(
            f"models:/{model_name}/{champion_version}"
        )
        champion_rmse, _ = score_model(
            champion_model,
            features,
            target,
        )

    reasons = []
    if champion_rmse is not None and candidate_rmse > champion_rmse:
        reasons.append("Candidate RMSE is worse than champion RMSE")
    if inference_ms >= threshold:
        reasons.append("Candidate inference time exceeds the limit")

    evaluation = {
        "candidate_rmse": candidate_rmse,
        "champion_rmse": champion_rmse,
        "champion_version": champion_version,
        "inference_ms_per_row": inference_ms,
        "inference_limit_ms_per_row": threshold,
        "test_rows": len(test),
        "eligible": not reasons,
        "reason": "; ".join(reasons) or "Promotion criteria passed",
    }
    result = {**candidate, "evaluation": evaluation}

    metrics = {
        "candidate_rmse": candidate_rmse,
        "inference_ms_per_row": inference_ms,
        "inference_limit_ms_per_row": threshold,
    }
    if champion_rmse is not None:
        metrics["champion_rmse"] = champion_rmse

    with mlflow.start_run(run_id=candidate["mlflow_run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "evaluation_status": (
                "passed" if evaluation["eligible"] else "rejected"
            ),
            "evaluated_champion_version": champion_version or "none",
            "evaluation_reason": evaluation["reason"],
        })
        mlflow.log_dict(result, "evaluated_candidate.json")

    return result

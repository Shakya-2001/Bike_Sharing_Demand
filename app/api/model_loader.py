from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    name: str
    version: str


def load_champion() -> LoadedModel:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ["MLFLOW_MODEL_NAME"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    champion = client.get_model_version_by_alias(
        model_name,
        "champion",
    )
    model = mlflow.sklearn.load_model(
        f"models:/{model_name}/{champion.version}"
    )

    sample = pd.DataFrame(
        [{
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": 8,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.24,
            "atemp": 0.28,
            "hum": 0.6,
            "windspeed": 0.2,
        }]
    )
    predictions = np.asarray(model.predict(sample), dtype=float)
    if predictions.shape != (1,) or not np.isfinite(predictions).all():
        raise ValueError("Champion failed its prediction readiness check")

    return LoadedModel(
        model=model,
        name=model_name,
        version=str(champion.version),
    )

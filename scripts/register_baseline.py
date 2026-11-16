from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from scripts.train_baseline import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ["MLFLOW_MODEL_NAME"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    registered_models = client.search_registered_models()
    existing = next(
        (model for model in registered_models if model.name == model_name),
        None,
    )
    if existing is not None and "champion" in existing.aliases:
        version = existing.aliases["champion"]
        print(f"Champion already exists: {model_name} version {version}")
        return

    model = joblib.load(
        PROJECT_ROOT / "models" / "bike_demand_model.pkl"
    )
    metadata = json.loads(
        (PROJECT_ROOT / "models" / "baseline_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    sample = pd.read_csv(
        PROJECT_ROOT / "demo_data" / "defense1" / "02_clean.csv"
    )[FEATURE_COLUMNS]
    expected_predictions = model.predict(sample)
    signature = infer_signature(sample, expected_predictions)

    mlflow.set_experiment("bike-demand-baseline")

    with mlflow.start_run(run_name="register-defense1-baseline"):
        mlflow.log_params(
            {
                "model_type": metadata["model_type"],
                "split": metadata["split"],
                "target": metadata["target"],
            }
        )
        mlflow.log_metrics(
            {name: metadata[name] for name in ("mae", "rmse", "r2")}
        )
        mlflow.set_tag(
            "metrics_source",
            "Stored Defense 1 evaluation, not recomputed during registration",
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=sample.head(2),
        )
        version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name,
        )

        loaded_model = mlflow.sklearn.load_model(
            f"models:/{model_name}/{version.version}"
        )
        np.testing.assert_allclose(
            loaded_model.predict(sample),
            expected_predictions,
        )

        client.set_registered_model_alias(
            model_name,
            "champion",
            version.version,
        )
        print(f"Registered {model_name} version {version.version}")
        print(f"Champion URI: models:/{model_name}@champion")
        print("Downloaded model predictions match the original model.")


if __name__ == "__main__":
    main()

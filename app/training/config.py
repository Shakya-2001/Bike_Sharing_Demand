"""Settings for the training job.

Everything that can change between machines is read from environment
variables when the function is called, so Docker and tests can override it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_NAME = os.getenv("MODEL_NAME", "bike_demand")
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bike_demand_training")

CATEGORICAL_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
NUMERIC_COLUMNS = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET_COLUMN = "cnt"

# The fixed test set: the most recent 20% of the dataset, never trained on.
TEST_FRACTION = 0.2


def get_training_config() -> dict[str, Any]:
    return {
        "dataset_path": os.getenv(
            "TRAINING_DATASET_PATH",
            str(DATA_DIR / "raw" / "hour.csv"),
        ),
        "good_data_dir": os.getenv(
            "TRAINING_GOOD_DATA_DIR",
            os.getenv("GOOD_DATA_DIR", str(DATA_DIR / "good_data")),
        ),
        "archived_data_dir": os.getenv(
            "ARCHIVED_DATA_DIR",
            str(DATA_DIR / "archived_data"),
        ),
        "state_file": os.getenv(
            "TRAINING_STATE_FILE",
            str(DATA_DIR / "training_state" / "last_training.json"),
        ),
        "min_new_rows": int(os.getenv("TRAINING_MIN_NEW_ROWS", "100")),
        "inference_threshold_ms": float(
            os.getenv("INFERENCE_TIME_THRESHOLD_MS", "100")
        ),
        "api_url": os.getenv("PREDICTION_API_URL", "http://api:8000"),
    }

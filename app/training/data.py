from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_baseline import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_COLUMNS = ["instant", "dteday", *FEATURE_COLUMNS, "cnt"]


def read_labeled_csv(path: Path) -> tuple[pd.DataFrame, str]:
    content = path.read_bytes()
    data = pd.read_csv(io.BytesIO(content))

    missing = [column for column in TRAINING_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    data = data[TRAINING_COLUMNS].copy()
    if data.empty or data.isna().any().any():
        raise ValueError(f"{path.name}: empty data or missing values")

    numeric_columns = ["instant", *FEATURE_COLUMNS, "cnt"]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{path.name}: non-finite numeric values")

    integer_columns = ["instant", *CATEGORICAL_COLUMNS, "cnt"]
    if (data[integer_columns] % 1 != 0).any().any():
        raise ValueError(f"{path.name}: fractional IDs, categories, or counts")
    if (data["cnt"] < 0).any() or (data["instant"] < 1).any():
        raise ValueError(f"{path.name}: negative target or invalid ID")
    if data["instant"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate IDs")

    pd.to_datetime(data["dteday"], errors="raise")
    digest = hashlib.sha256(content).hexdigest()
    return data, digest


def combine_training_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()

    if combined["instant"].duplicated().any():
        raise ValueError("Conflicting records share the same instant ID")

    return combined.sort_values("instant").reset_index(drop=True)


def load_training_data(min_new_rows: int = 100) -> dict:
    if min_new_rows < 1:
        raise ValueError("Minimum new rows must be positive")

    historical, _ = read_labeled_csv(
        DATA_DIR / "training" / "historical_train.csv"
    )
    test, test_digest = read_labeled_csv(
        DATA_DIR / "training" / "test.csv"
    )

    previous_frames = [historical]
    for path in sorted((DATA_DIR / "archived_data").glob("*.csv")):
        archived, _ = read_labeled_csv(path)
        previous_frames.append(archived)

    previous = combine_training_rows(previous_frames)
    incoming_frames = []
    files = []

    for path in sorted((DATA_DIR / "good_data").glob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns
        if "cnt" not in columns:
            continue

        incoming, digest = read_labeled_csv(path)
        incoming_frames.append(incoming)
        files.append({
            "path": str(path),
            "sha256": digest,
        })

    training_frames = [previous, *incoming_frames]
    training = combine_training_rows(training_frames)

    if training["instant"].max() >= test["instant"].min():
        raise ValueError("Training data overlaps or extends into the test set")

    new_rows = len(
        set(training["instant"]) - set(previous["instant"])
    )
    if new_rows < min_new_rows:
        return {
            "status": "insufficient_data",
            "new_rows": new_rows,
            "required_rows": min_new_rows,
        }

    return {
        "status": "ready",
        "new_rows": new_rows,
        "training": training,
        "test": test,
        "test_sha256": test_digest,
        "files": files,
    }

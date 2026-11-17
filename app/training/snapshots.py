from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app.training.data import (
    DATA_DIR,
    load_training_data,
    read_labeled_csv,
)


STATE_FILE = DATA_DIR / "training" / "last_training_attempt.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def create_training_snapshot(run_id: str) -> dict:
    run_key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    directory = DATA_DIR / "training_runs" / run_key
    manifest_path = directory / "manifest.json"

    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    required = int(os.getenv("TRAINING_MIN_NEW_ROWS", "100"))
    if required < 1:
        raise ValueError("Minimum new rows must be positive")

    prepared = load_training_data(min_new_rows=1)
    if prepared["status"] != "ready":
        return {
            "status": "insufficient_data",
            "new_rows": prepared["new_rows"],
            "required_rows": required,
        }

    new_rows = prepared["new_rows"]
    if STATE_FILE.exists():
        previous = json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )
        previous_ids = set(previous["training_ids"])
        current_ids = set(prepared["training"]["instant"])
        new_rows = len(current_ids - previous_ids)

    if new_rows < required:
        return {
            "status": "insufficient_data",
            "new_rows": new_rows,
            "required_rows": required,
        }

    directory.mkdir(parents=True, exist_ok=True)
    training_path = directory / "train.csv"
    test_path = directory / "test.csv"
    prepared["training"].to_csv(training_path, index=False)
    prepared["test"].to_csv(test_path, index=False)

    manifest = {
        "status": "ready",
        "run_id": run_id,
        "new_rows": new_rows,
        "training_path": str(training_path),
        "test_path": str(test_path),
        "training_sha256": hashlib.sha256(
            training_path.read_bytes()
        ).hexdigest(),
        "test_snapshot_sha256": hashlib.sha256(
            test_path.read_bytes()
        ).hexdigest(),
        "test_sha256": prepared["test_sha256"],
        "files": prepared["files"],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_training_snapshot(snapshot: dict) -> dict:
    training, training_digest = read_labeled_csv(
        Path(snapshot["training_path"])
    )
    test, test_digest = read_labeled_csv(
        Path(snapshot["test_path"])
    )
    if training_digest != snapshot["training_sha256"]:
        raise ValueError("Training snapshot changed")
    if test_digest != snapshot["test_snapshot_sha256"]:
        raise ValueError("Test snapshot changed")

    return {
        **snapshot,
        "training": training,
        "test": test,
    }


def record_training_attempt(snapshot: dict, candidate: dict) -> None:
    prepared = load_training_snapshot(snapshot)
    write_json_atomic(
        STATE_FILE,
        {
            "mlflow_run_id": candidate["mlflow_run_id"],
            "model_version": candidate["model_version"],
            "training_ids": (
                prepared["training"]["instant"].astype(int).tolist()
            ),
        },
    )

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.training.data import DATA_DIR


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_training_files(candidate: dict) -> dict:
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    from dags.prediction_dag import read_processed_files

    if candidate.get("promotion_status") != "promoted":
        raise ValueError("Archiving requires a successful promotion")

    version = str(int(candidate["model_version"]))
    hook = PostgresHook(
        postgres_conn_id=os.getenv(
            "AIRFLOW_POSTGRES_CONN_ID",
            "dsp_postgres",
        )
    )
    record = hook.get_first(
        """
        SELECT promoted_at
        FROM training_stats
        WHERE model_name = %s
          AND model_version = %s
          AND mlflow_run_id = %s
        """,
        parameters=(
            candidate["model_name"],
            version,
            candidate["mlflow_run_id"],
        ),
    )
    if record is None or record[0] is None:
        raise ValueError("Database has no matching successful promotion")

    good_dir = (DATA_DIR / "good_data").resolve()
    archive_dir = (DATA_DIR / "archived_data").resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    state_file = Path(
        os.getenv(
            "PREDICTION_STATE_FILE",
            str(DATA_DIR / "prediction_state" / "processed_files.json"),
        )
    )
    processed = read_processed_files(state_file)
    prediction_dir = Path(
        os.getenv("PREDICTION_GOOD_DATA_DIR", str(good_dir))
    )
    moves = []

    for item in candidate["files"]:
        filename = Path(item["path"]).name
        source = (good_dir / filename).resolve()
        destination = archive_dir / f"v{version}_{filename}"
        expected_digest = item["sha256"]

        if source.parent != good_dir:
            raise ValueError("Source must be inside good_data")

        if destination.exists():
            if file_digest(destination) != expected_digest:
                raise ValueError(f"Archive conflict for {filename}")

        if source.exists():
            if file_digest(source) != expected_digest:
                raise ValueError(f"Training file changed: {filename}")

            prediction_path = str(
                (prediction_dir / filename).resolve()
            )
            if prediction_path not in processed:
                raise RuntimeError(
                    f"Waiting for scheduled prediction of {filename}"
                )
        elif not destination.exists():
            raise FileNotFoundError(f"Training file missing: {filename}")

        moves.append((source, destination))

    for source, destination in moves:
        if source.exists():
            source.replace(destination)

    return {
        "status": "archived",
        "model_version": version,
        "files": len(moves),
    }

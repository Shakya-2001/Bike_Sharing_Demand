"""The steps of the training DAG, as plain functions.

The DAG file only calls these functions, which keeps them easy to test.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import requests

from app.training import registry
from app.training.alerts import build_rejection_message, send_ml_alert
from app.training.config import FEATURE_COLUMNS, MODEL_NAME, TEST_FRACTION
from app.training.data import (
    build_training_set,
    count_rows,
    find_good_files,
    load_full_dataset,
    read_trained_files,
    split_train_test,
    write_trained_files,
)
from app.training.evaluation import (
    decide_promotion,
    evaluate_model,
    measure_inference_ms,
)
from app.training.model import train_model
from app.training.stats import (
    compute_training_stats,
    get_hook,
    mark_baseline,
    save_training_stats,
)


def check_for_training_data(
    config: dict[str, Any],
    has_champion: bool,
) -> dict[str, Any]:
    """load_data: is there enough new data since the last training run?"""
    good_dir = Path(config["good_data_dir"])
    good_dir.mkdir(parents=True, exist_ok=True)
    all_files, new_files = find_good_files(
        good_dir,
        Path(config["state_file"]),
    )
    new_rows = count_rows(good_dir, new_files)
    minimum = int(config["min_new_rows"])

    if not has_champion:
        should_train = True
        reason = "No champion yet: training the first model"
    elif new_rows >= minimum:
        should_train = True
        reason = f"{new_rows} new rows (minimum is {minimum})"
    else:
        should_train = False
        reason = f"Only {new_rows} new rows (minimum is {minimum})"

    return {
        "should_train": should_train,
        "reason": reason,
        "good_files": all_files,
        "new_files": new_files,
        "new_rows": new_rows,
    }


def train_candidate(
    config: dict[str, Any],
    good_files: list[str],
) -> dict[str, Any]:
    """train_model: train, score on the test set, log and register."""
    full_df = load_full_dataset(Path(config["dataset_path"]))
    train_df, test_df, info = build_training_set(
        full_df,
        Path(config["good_data_dir"]),
        good_files,
    )
    model = train_model(train_df)
    metrics = evaluate_model(model, test_df)

    params = {
        **info,
        "model_type": "HistGradientBoostingRegressor",
        "test_fraction": TEST_FRACTION,
        "good_files_used": len(good_files),
    }
    registered = registry.log_and_register_candidate(
        model,
        params,
        {f"test_{name}": value for name, value in metrics.items()},
        train_df[FEATURE_COLUMNS].head(5),
    )
    return {
        **registered,
        "metrics": metrics,
        "info": info,
        "good_files": good_files,
        "training_stats": compute_training_stats(train_df),
    }


def save_candidate_stats(train_result: dict[str, Any]) -> dict[str, Any]:
    """save_training_stats: store the candidate's feature statistics."""
    saved = save_training_stats(
        get_hook(),
        train_result["training_stats"],
        MODEL_NAME,
        int(train_result["model_version"]),
        train_result["run_id"],
        int(train_result["info"]["train_rows"]),
    )
    return {"rows_saved": saved}


def evaluate_candidate(
    config: dict[str, Any],
    train_result: dict[str, Any],
) -> dict[str, Any]:
    """evaluate_candidate: same test set for candidate and champion."""
    full_df = load_full_dataset(Path(config["dataset_path"]))
    _, test_df = split_train_test(full_df)

    candidate = registry.load_model_version(train_result["model_version"])
    candidate_metrics = evaluate_model(candidate, test_df)
    inference_ms = measure_inference_ms(candidate, test_df)

    champion_version = registry.get_champion_version()
    champion_metrics = None
    if champion_version is not None:
        champion = registry.load_champion_model()
        champion_metrics = evaluate_model(champion, test_df)
    champion_rmse = (
        None if champion_metrics is None else champion_metrics["rmse"]
    )

    threshold_ms = float(config["inference_threshold_ms"])
    decision = decide_promotion(
        candidate_metrics["rmse"],
        champion_rmse,
        inference_ms,
        threshold_ms,
    )
    registry.log_comparison(
        train_result["run_id"],
        decision.promote,
        decision.reasons,
        None if champion_version is None else champion_version.version,
        champion_metrics,
        inference_ms,
        threshold_ms,
    )

    alert = None
    if not decision.promote:
        alert = send_ml_alert(
            build_rejection_message(
                train_result["model_version"],
                decision.reasons,
                candidate_metrics["rmse"],
                champion_rmse,
                inference_ms,
            )
        )

    state_file = Path(config["state_file"])
    write_trained_files(
        state_file,
        read_trained_files(state_file) | set(train_result["good_files"]),
    )
    return {
        "promote": decision.promote,
        "reasons": decision.reasons,
        "model_version": train_result["model_version"],
        "run_id": train_result["run_id"],
        "candidate_rmse": candidate_metrics["rmse"],
        "champion_rmse": champion_rmse,
        "inference_ms": inference_ms,
        "good_files": train_result["good_files"],
        "alert": alert,
    }


def promote_candidate(decision: dict[str, Any]) -> dict[str, Any]:
    """promote_to_champion: move @champion and set the drift baseline."""
    version = decision["model_version"]
    registry.promote_version(version)
    baseline_rows = mark_baseline(get_hook(), MODEL_NAME, int(version))
    return {
        "model_version": version,
        "baseline_rows": baseline_rows,
        "good_files": decision["good_files"],
    }


def reload_api(api_url: str) -> dict[str, Any]:
    """notify_api_reload: ask the API to load the new champion."""
    response = requests.post(
        f"{api_url.rstrip('/')}/reload-model",
        timeout=60,
    )
    response.raise_for_status()
    return {"status_code": response.status_code}


def archive_files(
    good_dir: Path,
    archived_dir: Path,
    file_names: list[str],
) -> list[str]:
    """archive_data: move the files used for training out of good_data."""
    archived_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in file_names:
        source = good_dir / name
        if source.exists():
            shutil.move(str(source), str(archived_dir / name))
            moved.append(name)
    return moved

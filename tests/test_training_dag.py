from __future__ import annotations

import pytest

from app.training import registry
from dags import training_dag
from tests.training_fixtures import FakeVersion, make_dataset, write_good_file


def test_training_dag_metadata() -> None:
    assert training_dag.DAG_ID == "bike_model_training"
    assert training_dag.MAX_ACTIVE_RUNS == 1
    assert training_dag.TASK_IDS == [
        "load_data",
        "train_model",
        "save_training_stats",
        "evaluate_candidate",
        "promote_to_champion",
        "notify_api_reload",
        "archive_data",
    ]


def set_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TRAINING_GOOD_DATA_DIR", str(tmp_path / "good"))
    monkeypatch.setenv("TRAINING_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("TRAINING_MIN_NEW_ROWS", "10")


def test_load_data_skips_the_run_when_not_enough_new_data(
    monkeypatch,
    tmp_path,
) -> None:
    set_paths(monkeypatch, tmp_path)
    write_good_file(tmp_path / "good", "a.csv", make_dataset(50), [1, 2])
    monkeypatch.setattr(
        registry,
        "get_champion_version",
        lambda: FakeVersion("1"),
    )

    with pytest.raises(training_dag.AirflowSkipException):
        training_dag.load_training_data()


def test_load_data_trains_the_first_model_without_a_champion(
    monkeypatch,
    tmp_path,
) -> None:
    set_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "get_champion_version", lambda: None)

    result = training_dag.load_training_data()

    assert result["should_train"] is True


def test_promote_is_skipped_when_the_candidate_was_rejected() -> None:
    decision = {"promote": False, "reasons": ["RMSE is worse"]}

    with pytest.raises(training_dag.AirflowSkipException):
        training_dag.run_promote(decision)

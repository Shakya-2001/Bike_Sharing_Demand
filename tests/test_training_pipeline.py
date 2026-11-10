from __future__ import annotations

from app.training import data, pipeline
from tests.training_fixtures import (
    FakeHook,
    FakeVersion,
    FixedModel,
    make_dataset,
    write_good_file,
)


def make_config(tmp_path, dataset_rows: int = 200) -> dict:
    dataset_path = tmp_path / "hour.csv"
    make_dataset(dataset_rows).to_csv(dataset_path, index=False)
    return {
        "dataset_path": str(dataset_path),
        "good_data_dir": str(tmp_path / "good"),
        "archived_data_dir": str(tmp_path / "archived"),
        "state_file": str(tmp_path / "state" / "last_training.json"),
        "min_new_rows": 10,
        "inference_threshold_ms": 100.0,
        "api_url": "http://api:8000",
    }


def test_train_candidate_logs_one_registered_model(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    write_good_file(
        tmp_path / "good",
        "a.csv",
        make_dataset(200),
        [5, 6, 190],
    )
    logged = {}

    def fake_log(model, params, metrics, input_example):
        logged["params"] = params
        logged["metrics"] = metrics
        return {"run_id": "run-1", "model_version": "2"}

    monkeypatch.setattr(
        pipeline.registry,
        "log_and_register_candidate",
        fake_log,
    )

    result = pipeline.train_candidate(config, ["a.csv"])

    assert result["run_id"] == "run-1"
    assert result["model_version"] == "2"
    assert result["info"]["train_rows"] == 160
    assert result["info"]["test_rows"] == 40
    assert set(logged["metrics"]) == {"test_mae", "test_rmse", "test_r2"}
    assert logged["params"]["new_rows_in_test_period"] == 1
    assert len(result["training_stats"]) > 0


def patch_registry(monkeypatch, champion_version, champion_model) -> None:
    monkeypatch.setattr(
        pipeline.registry,
        "load_model_version",
        lambda version: FixedModel(50),
    )
    monkeypatch.setattr(
        pipeline.registry,
        "get_champion_version",
        lambda: champion_version,
    )
    monkeypatch.setattr(
        pipeline.registry,
        "load_champion_model",
        lambda: champion_model,
    )
    monkeypatch.setattr(pipeline.registry, "log_comparison", lambda *a: None)


def make_train_result() -> dict:
    return {
        "run_id": "run-1",
        "model_version": "2",
        "good_files": ["a.csv", "b.csv"],
    }


def test_evaluate_promotes_first_model_and_marks_files_as_seen(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    patch_registry(monkeypatch, None, None)
    alerts = []
    monkeypatch.setattr(pipeline, "send_ml_alert", alerts.append)

    result = pipeline.evaluate_candidate(config, make_train_result())

    assert result["promote"] is True
    assert result["champion_rmse"] is None
    assert alerts == []
    seen = data.read_trained_files(tmp_path / "state" / "last_training.json")
    assert seen == {"a.csv", "b.csv"}


def test_evaluate_rejects_worse_candidate_and_sends_alert(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    test_rows = data.split_train_test(make_dataset(200))[1]
    perfect_champion = FixedModel(float(test_rows["cnt"].mean()))
    patch_registry(monkeypatch, FakeVersion("1"), perfect_champion)
    alerts = []
    monkeypatch.setattr(
        pipeline,
        "send_ml_alert",
        lambda message: alerts.append(message) or {"delivery_status": "ok"},
    )

    result = pipeline.evaluate_candidate(config, make_train_result())

    assert result["promote"] is False
    assert "worse" in result["reasons"][0]
    assert len(alerts) == 1
    assert "NOT promoted" in alerts[0]


def test_evaluate_rejects_a_candidate_that_is_too_slow(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    config["inference_threshold_ms"] = 0.0
    patch_registry(monkeypatch, FakeVersion("1"), FixedModel(50))
    monkeypatch.setattr(
        pipeline,
        "send_ml_alert",
        lambda message: {"delivery_status": "skipped"},
    )

    result = pipeline.evaluate_candidate(config, make_train_result())

    assert result["promote"] is False
    assert "Inference time" in result["reasons"][0]


def test_promote_moves_alias_and_sets_baseline(monkeypatch) -> None:
    calls = []
    hook = FakeHook(baseline_count=61)
    monkeypatch.setattr(
        pipeline.registry,
        "promote_version",
        lambda version: calls.append(version),
    )
    monkeypatch.setattr(pipeline, "get_hook", lambda: hook)

    result = pipeline.promote_candidate(
        {"model_version": "5", "good_files": ["a.csv"]}
    )

    assert calls == ["5"]
    assert result["baseline_rows"] == 61
    assert result["good_files"] == ["a.csv"]


def test_archive_moves_only_the_listed_files(tmp_path) -> None:
    dataset = make_dataset(20)
    good_dir = tmp_path / "good"
    archived_dir = tmp_path / "archived"
    write_good_file(good_dir, "a.csv", dataset, [1])
    write_good_file(good_dir, "b.csv", dataset, [2])

    moved = pipeline.archive_files(good_dir, archived_dir, ["a.csv"])

    assert moved == ["a.csv"]
    assert (archived_dir / "a.csv").exists()
    assert not (good_dir / "a.csv").exists()
    assert (good_dir / "b.csv").exists()


def test_reload_api_posts_to_the_reload_endpoint(monkeypatch) -> None:
    seen = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    def fake_post(url, timeout):
        seen["url"] = url
        return FakeResponse()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)

    result = pipeline.reload_api("http://api:8000/")

    assert seen["url"] == "http://api:8000/reload-model"
    assert result["status_code"] == 200

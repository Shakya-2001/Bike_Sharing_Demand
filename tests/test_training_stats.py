from __future__ import annotations

import pytest

from app.training import stats
from tests.training_fixtures import FakeHook, make_dataset


def test_numeric_stats_cover_features_and_target() -> None:
    result = stats.compute_training_stats(make_dataset(50))

    numeric = {
        item["feature_name"]: item
        for item in result
        if item["feature_type"] == "numeric"
    }
    assert set(numeric) == {"temp", "atemp", "hum", "windspeed", "cnt"}
    assert numeric["temp"]["mean"] == pytest.approx(0.5)
    assert numeric["cnt"]["min_value"] == 10.0
    assert numeric["cnt"]["max_value"] == 59.0


def test_categorical_shares_add_up_to_one() -> None:
    result = stats.compute_training_stats(make_dataset(48))

    hours = [item for item in result if item["feature_name"] == "hr"]
    assert len(hours) == 24
    assert sum(item["proportion"] for item in hours) == pytest.approx(1.0)
    assert all(item["category"] is not None for item in hours)


def test_save_training_stats_inserts_rows_that_are_not_baseline() -> None:
    hook = FakeHook()
    computed = stats.compute_training_stats(make_dataset(48))

    saved = stats.save_training_stats(
        hook,
        computed,
        "bike_demand",
        3,
        "run-abc",
        48,
    )

    assert saved == len(computed)
    inserted = hook.inserted[0]
    assert inserted["table"] == "training_stats"
    first_row = dict(zip(inserted["fields"], inserted["rows"][0]))
    assert first_row["model_version"] == 3
    assert first_row["run_id"] == "run-abc"
    assert first_row["is_baseline"] is False


def test_save_training_stats_replaces_earlier_rows_of_the_same_run() -> None:
    hook = FakeHook()

    stats.save_training_stats(hook, [], "bike_demand", 1, "run-abc", 10)

    single = [sql for sql, _ in hook.runs if isinstance(sql, str)]
    delete_calls = [sql for sql in single if sql.startswith("DELETE")]
    assert len(delete_calls) == 1


def test_mark_baseline_clears_old_baseline_then_sets_new_one() -> None:
    hook = FakeHook(baseline_count=61)

    count = stats.mark_baseline(hook, "bike_demand", 4)

    updates = [
        (sql, parameters)
        for sql, parameters in hook.runs
        if isinstance(sql, str) and sql.startswith("UPDATE")
    ]
    assert "FALSE" in updates[0][0]
    assert "TRUE" in updates[1][0]
    assert updates[1][1] == ("bike_demand", 4)
    assert count == 61

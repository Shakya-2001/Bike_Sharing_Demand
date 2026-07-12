from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from app.web import api_client
from app.web.prediction_data import FEATURE_COLUMNS, SEASON_MONTHS
from app.web.prediction_data import SOURCE_NAMES, WEEKDAY_NAMES, batch_rows
from app.web.prediction_data import history_frame, prediction_frame
from app.web.prediction_data import normalize_feels_like_temperature
from app.web.prediction_data import normalize_humidity, normalize_temperature
from app.web.prediction_data import normalize_wind_speed


def sample_features() -> dict:
    return {
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
    }


def test_streamlit_has_two_pages() -> None:
    assert Path("app/web/Home.py").exists()
    assert Path("app/web/pages/2_Past_Predictions.py").exists()


def test_api_client_defaults_to_local_fastapi() -> None:
    assert api_client.API_BASE_URL == "http://127.0.0.1:8000"
    assert api_client.REQUEST_TIMEOUT == 10


def test_season_month_options_are_consistent() -> None:
    assert 12 not in SEASON_MONTHS[2]
    assert SEASON_MONTHS[1] == [3, 4, 5]
    assert SEASON_MONTHS[2] == [6, 7, 8]
    assert SEASON_MONTHS[3] == [9, 10, 11]
    assert SEASON_MONTHS[4] == [12, 1, 2]


def test_weekday_names_match_model_codes() -> None:
    assert WEEKDAY_NAMES == {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
    }


def test_prediction_source_names_are_user_friendly() -> None:
    assert SOURCE_NAMES == {
        "all": "All predictions",
        "webapp": "Manual predictions",
        "scheduled": "Scheduled predictions",
    }


def test_celsius_temperatures_are_normalized_for_model() -> None:
    assert normalize_temperature(-8.0) == 0.0
    assert normalize_temperature(39.0) == 1.0
    assert normalize_feels_like_temperature(-16.0) == 0.0
    assert normalize_feels_like_temperature(50.0) == 1.0
    assert normalize_temperature(18.0) == pytest.approx(0.5532, abs=0.0001)


def test_weather_units_are_normalized_for_model() -> None:
    assert normalize_humidity(0) == 0.0
    assert normalize_humidity(60) == 0.6
    assert normalize_humidity(100) == 1.0
    assert normalize_wind_speed(0.0) == 0.0
    assert normalize_wind_speed(67.0) == 1.0
    assert normalize_wind_speed(20.0) == pytest.approx(0.2985, abs=0.0001)


def test_batch_rows_requires_all_features() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        batch_rows(pd.DataFrame([{"season": 1}]))


def test_batch_rows_ignores_labels_and_preserves_samples() -> None:
    first = sample_features()
    second = {**first, "hr": 9}
    data = pd.DataFrame([{**first, "cnt": 100}, {**second, "cnt": 120}])

    rows = batch_rows(data)

    assert len(rows) == 2
    assert list(rows[0]) == FEATURE_COLUMNS
    assert "cnt" not in rows[0]


def test_predict_demand_sends_batch_in_request_body(monkeypatch) -> None:
    response = Mock()
    response.json.return_value = {"predictions": []}
    post = Mock(return_value=response)
    monkeypatch.setattr(api_client.requests, "post", post)
    rows = [sample_features(), {**sample_features(), "hr": 9}]

    api_client.predict_demand(rows)

    _, kwargs = api_client.requests.post.call_args
    assert kwargs["json"] == {"features": rows, "source": "webapp"}


def test_history_request_sends_filters(monkeypatch) -> None:
    response = Mock()
    response.json.return_value = {"predictions": []}
    get = Mock(return_value=response)
    monkeypatch.setattr(api_client.requests, "get", get)

    api_client.get_past_predictions(
        source="scheduled",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 12),
        limit=50,
    )

    _, kwargs = api_client.requests.get.call_args
    assert kwargs["params"] == {
        "source": "scheduled",
        "start_date": "2026-07-01T00:00:00",
        "end_date": "2026-07-12T23:59:59",
        "limit": 50,
    }


def test_results_include_features_and_predictions() -> None:
    features = sample_features()
    result = {"predictions": [{"features": features, "prediction": 42.5}]}
    history = [{
        "features": features,
        "prediction": 42.5,
        "source": "webapp",
        "model_name": "model-v1",
        "created_at": "2026-07-12T10:00:00",
    }]

    assert prediction_frame(result).iloc[0]["prediction"] == 42.5
    assert history_frame(history).iloc[0]["source"] == "webapp"


def test_webapp_does_not_import_model_or_database() -> None:
    for path in Path("app/web").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "joblib" not in source
        assert "app.database" not in source

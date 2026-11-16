from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.api.main as api
from app.api.model_loader import LoadedModel


PAYLOAD = {
    "features": [{
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
    }],
    "source": "webapp",
}


@pytest.fixture
def client(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    model = joblib.load(
        project_root / "models" / "bike_demand_model.pkl"
    )
    champion = LoadedModel(
        model=model,
        name="BikeDemand",
        version="1",
    )
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    def get_test_db():
        with Session(test_engine) as session:
            yield session

    monkeypatch.setattr(api, "engine", test_engine)
    monkeypatch.setattr(api, "load_champion", lambda: champion)

    previous_overrides = api.app.dependency_overrides.copy()
    api.app.dependency_overrides[api.get_db] = get_test_db
    try:
        with TestClient(api.app) as test_client:
            yield test_client
    finally:
        api.app.dependency_overrides.clear()
        api.app.dependency_overrides.update(previous_overrides)
        test_engine.dispose()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_records_model_version(client):
    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "BikeDemand"
    assert body["model_version"] == "1"
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["prediction"] > 0

    history = client.get("/past-predictions").json()["predictions"]
    assert len(history) == 1
    assert history[0]["model_name"] == "BikeDemand"
    assert history[0]["model_version"] == "1"


def test_reload_updates_prediction_version(client, monkeypatch):
    next_champion = replace(api.active_model, version="2")
    monkeypatch.setattr(api, "load_champion", lambda: next_champion)

    response = client.post("/reload-model")

    assert response.status_code == 200
    assert response.json()["model_version"] == "2"

    prediction = client.post("/predict", json=PAYLOAD)
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == "2"

    history = client.get("/past-predictions").json()["predictions"]
    assert history[0]["model_version"] == "2"


def test_failed_reload_preserves_predictions(client, monkeypatch):
    previous_model = api.active_model

    def fail_loading():
        raise RuntimeError("Registry unavailable")

    monkeypatch.setattr(api, "load_champion", fail_loading)

    response = client.post("/reload-model")

    assert response.status_code == 503
    assert api.active_model is previous_model

    prediction = client.post("/predict", json=PAYLOAD)
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == "1"

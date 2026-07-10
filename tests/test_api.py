from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_returns_prediction() -> None:
    payload = {
        "features": [
            {
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
        ],
        "source": "webapp",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "bike_demand_hist_gradient_boosting_v1"
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["prediction"] > 0

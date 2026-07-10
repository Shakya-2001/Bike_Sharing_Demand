from __future__ import annotations

import os
from typing import Any

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10


class ApiClientError(RuntimeError):
    pass


def predict_demand(features: dict[str, Any], source: str = "webapp") -> dict:
    payload = {
        "features": [features],
        "source": source,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiClientError(
            "Prediction API is not available. Start FastAPI first."
        ) from exc

    return response.json()


def get_past_predictions(limit: int = 20) -> list[dict]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/past-predictions",
            params={"limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiClientError(
            "Prediction history is not available. Start FastAPI first."
        ) from exc

    return response.json().get("predictions", [])

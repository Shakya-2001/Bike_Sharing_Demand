from __future__ import annotations

import os
from datetime import date
from typing import Any

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10


class ApiClientError(RuntimeError):
    pass


def predict_demand(
    features: dict[str, Any] | list[dict[str, Any]],
    source: str = "webapp",
) -> dict:
    rows = features if isinstance(features, list) else [features]
    payload = {
        "features": rows,
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


def get_past_predictions(
    source: str = "all",
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 100,
) -> list[dict]:
    params: dict[str, str | int] = {
        "source": source,
        "limit": limit,
    }
    if start_date is not None:
        params["start_date"] = f"{start_date.isoformat()}T00:00:00"
    if end_date is not None:
        params["end_date"] = f"{end_date.isoformat()}T23:59:59"

    try:
        response = requests.get(
            f"{API_BASE_URL}/past-predictions",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiClientError(
            "Prediction history is not available. Start FastAPI first."
        ) from exc

    return response.json().get("predictions", [])

from __future__ import annotations

import os

import requests


def notify_api_reload(candidate: dict) -> dict:
    if candidate.get("promotion_status") != "promoted":
        raise ValueError("API reload requires a successful promotion")

    api_url = os.getenv(
        "PREDICTION_API_URL",
        "http://api:8000",
    ).rstrip("/")

    response = requests.post(
        f"{api_url}/reload-model",
        timeout=60,
    )
    response.raise_for_status()
    loaded = response.json()

    if (
        loaded.get("model_name") != candidate["model_name"]
        or loaded.get("model_version") != candidate["model_version"]
    ):
        raise RuntimeError("API loaded a different model version")

    return {
        "status": "loaded",
        "model_name": loaded["model_name"],
        "model_version": loaded["model_version"],
    }

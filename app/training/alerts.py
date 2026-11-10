"""Teams alerts for the ML Alerts channel."""
from __future__ import annotations

import os
from typing import Any

import requests


def build_rejection_message(
    model_version: str,
    reasons: list[str],
    candidate_rmse: float,
    champion_rmse: float | None,
    inference_ms: float,
) -> str:
    champion_text = (
        "none" if champion_rmse is None else f"{champion_rmse:.2f}"
    )
    return (
        f"Candidate model version {model_version} was NOT promoted. "
        f"Reasons: {'; '.join(reasons)}. "
        f"Candidate RMSE {candidate_rmse:.2f}, champion RMSE "
        f"{champion_text}, inference time {inference_ms:.3f} ms."
    )


def send_ml_alert(message: str) -> dict[str, Any]:
    """Post to the ML Alerts Teams channel (webhook from the environment)."""
    webhook_url = os.getenv("ML_ALERTS_WEBHOOK_URL")
    if not webhook_url:
        return {
            "delivery_status": "skipped",
            "delivery_reason": "No ML_ALERTS_WEBHOOK_URL configured",
        }
    try:
        response = requests.post(
            webhook_url,
            json={
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Bike sharing ML alert",
                "themeColor": "E81123",
                "title": "Bike Sharing ML Alert",
                "text": message,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"delivery_status": "failed", "delivery_reason": str(exc)}
    return {"delivery_status": "sent", "status_code": response.status_code}

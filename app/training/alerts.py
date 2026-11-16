from __future__ import annotations

import os

import requests


def send_rejection_alert(candidate: dict) -> dict:
    evaluation = candidate["evaluation"]
    if evaluation["eligible"]:
        raise ValueError("Rejection alerts require a rejected candidate")

    webhook_url = os.environ["TEAMS_ML_WEBHOOK_URL"]
    if not webhook_url.strip():
        raise ValueError("ML Alerts webhook is empty")

    message = (
        f"Model: {candidate['model_name']}\n\n"
        f"Candidate version: {candidate['model_version']}\n\n"
        f"Reason: {evaluation['reason']}\n\n"
        f"Candidate RMSE: "
        f"{evaluation.get('candidate_rmse', 'not measured')}\n\n"
        f"Champion RMSE: "
        f"{evaluation.get('champion_rmse', 'not measured')}\n\n"
        f"Inference milliseconds per row: "
        f"{evaluation.get('inference_ms_per_row', 'not measured')}"
    )

    response = requests.post(
        webhook_url,
        json={
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": "Bike demand candidate rejection",
            "themeColor": "E81123",
            "title": "Bike Demand Candidate Rejected",
            "text": message,
        },
        timeout=15,
    )
    response.raise_for_status()

    return {
        "status": "sent",
        "model_version": candidate["model_version"],
    }

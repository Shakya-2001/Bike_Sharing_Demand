"""Scoring and promotion rules for the training job."""
from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.training.config import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str]


def evaluate_model(model: Any, test_df: pd.DataFrame) -> dict[str, float]:
    """Score a model on the fixed test set (lower mae/rmse is better)."""
    predictions = model.predict(test_df[FEATURE_COLUMNS])
    actual = test_df[TARGET_COLUMN]
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
        "r2": float(r2_score(actual, predictions)),
    }


def measure_inference_ms(
    model: Any,
    test_df: pd.DataFrame,
    repeats: int = 30,
) -> float:
    """Median time to predict ONE row, like a single API request."""
    row = test_df[FEATURE_COLUMNS].iloc[[0]]
    model.predict(row)  # warm-up call, not measured
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(row)
        timings.append((time.perf_counter() - start) * 1000)
    return float(median(timings))


def decide_promotion(
    candidate_rmse: float,
    champion_rmse: float | None,
    inference_ms: float,
    threshold_ms: float,
) -> PromotionDecision:
    """Promote if at least as accurate as the champion and fast enough.

    Primary metric: RMSE (lower is better). A tie is enough to promote.
    With no champion yet (first run) the candidate is promoted.
    """
    if champion_rmse is None:
        return PromotionDecision(
            True,
            ["No champion exists yet: the first model is promoted"],
        )

    reasons = []
    if candidate_rmse > champion_rmse:
        reasons.append(
            f"RMSE {candidate_rmse:.2f} is worse than the champion's "
            f"{champion_rmse:.2f}"
        )
    if inference_ms >= threshold_ms:
        reasons.append(
            f"Inference time {inference_ms:.3f} ms is not below the "
            f"{threshold_ms:g} ms threshold"
        )
    if reasons:
        return PromotionDecision(False, reasons)
    return PromotionDecision(
        True,
        ["Candidate matches or beats the champion and is fast enough"],
    )

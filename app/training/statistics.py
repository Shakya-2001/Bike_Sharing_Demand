from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_baseline import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
)


def calculate_feature_statistics(
    data: pd.DataFrame,
) -> tuple[dict, dict]:
    if data.empty:
        raise ValueError("Cannot calculate statistics for empty data")

    features = data[FEATURE_COLUMNS]
    if features.isna().any().any():
        raise ValueError("Training features contain missing values")

    numerical_stats = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {column}")

        numerical_stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(values.quantile(0.05)),
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
        }

    categorical_stats = {}
    for column in CATEGORICAL_COLUMNS:
        proportions = features[column].value_counts(normalize=True)
        categorical_stats[column] = {
            str(category): float(proportion)
            for category, proportion in proportions.items()
        }

    return numerical_stats, categorical_stats

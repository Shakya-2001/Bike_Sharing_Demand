"""The model that the training job trains.

This is the same pipeline as the Defense 1 baseline
(scripts/train_baseline.py), so a retrained model is comparable with it.
It is copied here because the Airflow image only contains the `app` folder.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from app.training.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=float("nan"),
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
    )
    regressor = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL_COLUMNS))),
        learning_rate=0.1,
        max_iter=600,
        max_leaf_nodes=64,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_model(train_df: pd.DataFrame) -> Pipeline:
    model = build_model()
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return model

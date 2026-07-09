import json

import joblib
import pandas as pd

from scripts import train_baseline


PROJECT_ROOT = train_baseline.PROJECT_ROOT
EXPECTED_METRICS = {
    "model_type",
    "rows",
    "features",
    "target",
    "mae",
    "rmse",
    "r2",
}


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 1,
                "yr": 0,
                "mnth": 1,
                "hr": 8,
                "holiday": 0,
                "weekday": 6,
                "workingday": 0,
                "weathersit": 1,
                "temp": 0.24,
                "atemp": 0.29,
                "hum": 0.81,
                "windspeed": 0.0,
            }
        ]
    )


def test_training_configuration_prevents_target_leakage() -> None:
    assert train_baseline.TARGET_COLUMN == "cnt"
    assert "casual" not in train_baseline.FEATURE_COLUMNS
    assert "registered" not in train_baseline.FEATURE_COLUMNS
    assert set(sample_features().columns) == set(
        train_baseline.FEATURE_COLUMNS
    )


def test_committed_model_and_metrics_are_usable() -> None:
    model_path = PROJECT_ROOT / "models" / "bike_demand_model.pkl"
    metrics_path = PROJECT_ROOT / "models" / "baseline_metrics.json"

    model = joblib.load(model_path)
    prediction = model.predict(sample_features())
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert len(prediction) == 1
    assert float(prediction[0]) >= 0
    assert EXPECTED_METRICS <= set(metrics)
    assert metrics["target"] == train_baseline.TARGET_COLUMN
    assert metrics["features"] == train_baseline.FEATURE_COLUMNS


def test_training_creates_model_and_metrics(tmp_path, monkeypatch) -> None:
    rows = []
    for index in range(100):
        row = sample_features().iloc[0].to_dict()
        row["hr"] = index % 24
        row["weekday"] = index % 7
        row["temp"] = 0.2 + (index % 20) / 100
        row["cnt"] = 20 + row["hr"] * 4 + index % 10
        rows.append(row)

    data_path = tmp_path / "hour.csv"
    model_path = tmp_path / "bike_demand_model.pkl"
    metrics_path = tmp_path / "baseline_metrics.json"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    monkeypatch.setattr(train_baseline, "DATA_PATH", data_path)
    monkeypatch.setattr(train_baseline, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train_baseline, "MODEL_PATH", model_path)
    monkeypatch.setattr(train_baseline, "METRICS_PATH", metrics_path)

    train_baseline.main()

    assert model_path.exists()
    assert metrics_path.exists()
    generated_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert generated_metrics["rows"] == 100
    assert generated_metrics["target"] == "cnt"

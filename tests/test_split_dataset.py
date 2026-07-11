from __future__ import annotations

import pandas as pd

from scripts.split_dataset import DEFAULT_COLUMNS, ROWS_PER_FILE, split_dataset


def test_split_dataset_creates_exactly_10_rows_per_file(tmp_path) -> None:
    dataset_path = tmp_path / "input.csv"
    output_dir = tmp_path / "raw_data"
    rows = []
    for index in range(30):
        rows.append(
            {
                "instant": index + 1,
                "dteday": "2012-01-01",
                "season": 1,
                "yr": 1,
                "mnth": 1,
                "hr": index % 24,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.4,
                "atemp": 0.4,
                "hum": 0.5,
                "windspeed": 0.2,
                "casual": 1,
                "registered": 2,
                "cnt": 3,
            }
        )
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    generated_files = split_dataset(
        dataset_path,
        output_dir,
        number_of_files=3,
    )

    assert len(generated_files) == 3
    for generated_file in generated_files:
        batch = pd.read_csv(generated_file)
        assert len(batch) == ROWS_PER_FILE
        assert list(batch.columns) == DEFAULT_COLUMNS

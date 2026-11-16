from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.train_baseline import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROWS_PER_FILE = 10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare labeled training, incoming, and test data."
    )
    parser.add_argument("--number-of-files", type=int, default=30)
    args = parser.parse_args()

    if args.number_of_files < 1:
        raise ValueError("Number of files must be positive")

    source = PROJECT_ROOT / "data" / "raw" / "hour.csv"
    output_dir = PROJECT_ROOT / "data" / "training"
    batch_dir = output_dir / "incoming_batches"

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Review it before preparing again."
        )

    columns = ["instant", "dteday", *FEATURE_COLUMNS, "cnt"]
    data = pd.read_csv(source)
    data = data.sort_values("instant").reset_index(drop=True)
    data = data[columns]

    if data.isna().any().any():
        raise ValueError("Source data contains missing values")
    if data["instant"].duplicated().any():
        raise ValueError("Source data contains duplicate IDs")

    train_end = int(len(data) * 0.8)
    incoming_rows = args.number_of_files * ROWS_PER_FILE
    history_end = train_end - incoming_rows

    if history_end < 1:
        raise ValueError("Not enough rows for history and incoming batches")

    historical = data.iloc[:history_end]
    incoming = data.iloc[history_end:train_end]
    test = data.iloc[train_end:]

    batch_dir.mkdir(parents=True)
    historical.to_csv(output_dir / "historical_train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    for index in range(args.number_of_files):
        start = index * ROWS_PER_FILE
        batch = incoming.iloc[start:start + ROWS_PER_FILE]
        batch.to_csv(
            batch_dir / f"labeled_{index + 1:03d}.csv",
            index=False,
        )

    print(f"Historical training rows: {len(historical)}")
    print(f"Incoming labeled rows: {len(incoming)}")
    print(f"Test rows: {len(test)}")
    print(
        f"Created {args.number_of_files} batches, "
        f"{ROWS_PER_FILE} rows each"
    )
    print("Targets retained. casual and registered excluded.")
    print("Incoming batches are outside raw_data.")


if __name__ == "__main__":
    main()

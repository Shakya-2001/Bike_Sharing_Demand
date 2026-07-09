from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike+sharing+dataset.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "bike_sharing_dataset.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"Downloading dataset to {ZIP_PATH}")
        urlretrieve(DATASET_URL, ZIP_PATH)
    else:
        print(f"Dataset archive already exists at {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW_DIR)

    expected_file = RAW_DIR / "hour.csv"
    if not expected_file.exists():
        raise FileNotFoundError(
            f"Expected {expected_file} after extracting dataset"
        )

    print(f"Dataset ready: {expected_file}")


if __name__ == "__main__":
    main()

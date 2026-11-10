"""Dataset helpers for the training job.

Labels: the ingested files have no `cnt` column, so the real rental count of
each ingested row is looked up in the original dataset by its `instant`.

Test set: the most recent 20% of the dataset (by `instant`), the same split
used by the baseline model. It is never used for training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from app.training.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_FRACTION,
)

FEATURE_FILE_COLUMNS = ["instant"] + FEATURE_COLUMNS
DATASET_COLUMNS = FEATURE_FILE_COLUMNS + [TARGET_COLUMN]


def load_full_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run scripts/download_dataset.py first."
        )
    df = pd.read_csv(path).sort_values("instant")
    return df[DATASET_COLUMNS].reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last rows are the fixed test set."""
    n_test = max(1, math.ceil(len(df) * test_fraction))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def read_good_files(good_dir: Path, file_names: list[str]) -> pd.DataFrame:
    frames = [
        pd.read_csv(good_dir / name)[FEATURE_FILE_COLUMNS]
        for name in file_names
    ]
    if not frames:
        return pd.DataFrame(columns=FEATURE_FILE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def label_new_rows(
    new_rows: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the real `cnt` to ingested rows, looked up by `instant`."""
    if new_rows.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)
    labels = full_df[["instant", TARGET_COLUMN]]
    return new_rows.merge(labels, on="instant", how="inner")


def build_training_set(
    full_df: pd.DataFrame,
    good_dir: Path,
    file_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Original training part + labeled good_data rows, minus the test set.

    Ingested rows are copies of dataset rows, so rows already in the original
    training part are dropped (deduplicated by `instant`).
    """
    base_train, test = split_train_test(full_df)
    new_rows = read_good_files(good_dir, file_names)
    labeled = label_new_rows(new_rows, full_df)

    last_train_instant = int(base_train["instant"].max())
    usable = labeled[labeled["instant"] <= last_train_instant]
    train = pd.concat(
        [base_train, usable[DATASET_COLUMNS]],
        ignore_index=True,
    ).drop_duplicates(subset="instant", keep="first")

    info = {
        "base_train_rows": int(len(base_train)),
        "new_rows_read": int(len(new_rows)),
        "new_rows_labeled": int(len(labeled)),
        "new_rows_in_test_period": int(len(labeled) - len(usable)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    return train.reset_index(drop=True), test, info


def read_trained_files(state_file: Path) -> set[str]:
    """Names of the good_data files already seen by a training run."""
    if not state_file.exists():
        return set()
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Cannot read training state: {exc}") from exc
    return set(payload.get("trained_files", []))


def write_trained_files(state_file: Path, file_names: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = state_file.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps({"trained_files": sorted(file_names)}, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(state_file)


def find_good_files(
    good_dir: Path,
    state_file: Path,
) -> tuple[list[str], list[str]]:
    """Return (all good_data files, files not seen by a training run)."""
    all_files = sorted(path.name for path in good_dir.glob("*.csv"))
    trained = read_trained_files(state_file)
    new_files = [name for name in all_files if name not in trained]
    return all_files, new_files


def count_rows(good_dir: Path, file_names: list[str]) -> int:
    return sum(len(pd.read_csv(good_dir / name)) for name in file_names)

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.validation.bike_data import ValidationSummary, validate_bike_data

if TYPE_CHECKING:
    from app.database.models import IngestionStat


def choose_random_raw_file(raw_data_dir: Path) -> Path | None:
    candidates = sorted(raw_data_dir.glob("*.csv"))
    if not candidates:
        return None
    return random.choice(candidates)


def save_statistics(
    db: Session,
    summary: ValidationSummary,
) -> "IngestionStat":
    from app.database.models import IngestionStat

    stat = IngestionStat(
        file_name=Path(summary.file_path).name,
        total_rows=summary.total_rows,
        valid_rows=summary.valid_rows,
        invalid_rows=summary.invalid_rows,
        invalid_percent=summary.invalid_percent,
        criticality=summary.criticality,
        missing_columns=summary.missing_columns,
        error_counts=summary.error_counts,
        gx_success=summary.gx_success,
        gx_failed_expectations=summary.gx_failed_expectations,
    )
    db.add(stat)
    db.commit()
    db.refresh(stat)
    return stat


def split_and_save_data(
    source_file: Path,
    summary: ValidationSummary,
    good_df,
    bad_df,
    good_data_dir: Path,
    bad_data_dir: Path,
) -> dict[str, str | None]:
    good_data_dir.mkdir(parents=True, exist_ok=True)
    bad_data_dir.mkdir(parents=True, exist_ok=True)

    stem = source_file.stem
    outputs: dict[str, str | None] = {"good_file": None, "bad_file": None}

    if summary.invalid_rows == 0:
        good_path = good_data_dir / f"{stem}_good.csv"
        shutil.move(str(source_file), good_path)
        outputs["good_file"] = str(good_path)
        return outputs

    if summary.valid_rows == 0:
        bad_path = bad_data_dir / f"{stem}_bad.csv"
        shutil.move(str(source_file), bad_path)
        outputs["bad_file"] = str(bad_path)
        return outputs

    good_path = good_data_dir / f"{stem}_good.csv"
    bad_path = bad_data_dir / f"{stem}_bad.csv"
    good_df.to_csv(good_path, index=False)
    bad_df.to_csv(bad_path, index=False)
    source_file.unlink()
    outputs["good_file"] = str(good_path)
    outputs["bad_file"] = str(bad_path)
    return outputs


def ingest_file(
    file_path: Path,
    good_data_dir: Path,
    bad_data_dir: Path,
    db: Session | None = None,
) -> tuple[ValidationSummary, dict[str, str | None]]:
    summary, good_df, bad_df = validate_bike_data(file_path)
    if db is not None:
        save_statistics(db, summary)
    outputs = split_and_save_data(
        source_file=file_path,
        summary=summary,
        good_df=good_df,
        bad_df=bad_df,
        good_data_dir=good_data_dir,
        bad_data_dir=bad_data_dir,
    )
    return summary, outputs

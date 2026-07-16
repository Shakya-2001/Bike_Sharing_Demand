from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest one random raw CSV file."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_data",
    )
    parser.add_argument(
        "--good-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "good_data",
    )
    parser.add_argument(
        "--bad-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "bad_data",
    )
    return parser.parse_args()


def main() -> None:
    from app.database.models import Base
    from app.database.session import SessionLocal, engine
    from app.ingestion.pipeline import choose_random_raw_file, ingest_file

    args = parse_args()
    selected_file = choose_random_raw_file(args.raw_data_dir)
    if selected_file is None:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "No raw data files found",
                }
            )
        )
        return

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        summary, outputs = ingest_file(
            file_path=selected_file,
            good_data_dir=args.good_data_dir,
            bad_data_dir=args.bad_data_dir,
            db=db,
        )

    print(
        json.dumps(
            {
                "status": "processed",
                "summary": summary.to_dict(),
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

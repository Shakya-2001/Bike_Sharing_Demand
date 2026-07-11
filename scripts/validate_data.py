from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one bike-sharing CSV file."
    )
    parser.add_argument("file_path", type=Path)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    from app.validation.bike_data import validate_bike_data

    args = parse_args()
    summary, _, _ = validate_bike_data(args.file_path)
    payload = summary.to_dict()

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

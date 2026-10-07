"""
validate.py

Purpose:
    Validate the cleaned water consumption dataset.

Input:
    Cleaned Parquet dataset.

Output:
    Validation report.

This module validates data quality.
It does not perform cleaning.
"""

from pathlib import Path
import argparse
import logging

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path(
    "data/processed/cleaned_water_data.parquet"
)

DEFAULT_REPORT_PATH = Path(
    "reports/validation/validation_report.txt"
)


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Expected columns
# -------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "date",
    "hour",
    "day_of_week",
    "day_name",
    "month",
    "year",
    "sensor_id",
    "apartment",
    "cluster_name",
    "room_number",
    "room_type",
    "type",
    "attached_to",
    "value",
    "aggregated_value",
]


# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------

def load_data(
    input_path: Path,
) -> pd.DataFrame:
    """Load cleaned Parquet data."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}"
        )

    logger.info(
        "Loading cleaned dataset: %s",
        input_path,
    )

    return pd.read_parquet(input_path)


def check_required_columns(
    df: pd.DataFrame,
) -> None:
    """Check that all required columns exist."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info(
        "Required column check passed."
    )


def build_schema() -> DataFrameSchema:
    """Build Pandera validation schema."""

    return DataFrameSchema(
        {
            "timestamp": Column(
                pa.DateTime,
                nullable=False,
            ),

            "value": Column(
                float,
                checks=Check.ge(0),
                nullable=False,
            ),

            "aggregated_value": Column(
                float,
                checks=Check.ge(0),
                nullable=False,
            ),
        },
        strict=False,
    )


def validate_schema(
    df: pd.DataFrame,
) -> None:
    """Validate the dataset using Pandera."""

    schema = build_schema()

    schema.validate(
        df,
        lazy=True,
    )

    logger.info(
        "Pandera schema validation passed."
    )


def validate_timestamp_consistency(
    df: pd.DataFrame,
) -> None:
    """Check timestamp-derived columns."""

    timestamp = df["timestamp"]

    checks = {
        "hour": timestamp.dt.hour,
        "day_of_week": timestamp.dt.dayofweek,
        "month": timestamp.dt.month,
        "year": timestamp.dt.year,
    }

    for column, expected in checks.items():

        mismatch_count = int(
            (df[column].to_numpy() != expected.to_numpy()).sum()
        )

        if mismatch_count > 0:
            raise ValueError(
                f"Timestamp consistency check failed for "
                f"{column}: {mismatch_count} mismatches"
            )

    logger.info(
        "Timestamp consistency validation passed."
    )


def generate_report(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write validation summary."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = [
        "DATA VALIDATION REPORT",
        "=" * 60,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        f"Minimum value: {df['value'].min()}",
        f"Maximum value: {df['value'].max()}",
        "",
        "Validation status: PASSED",
    ]

    output_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    logger.info(
        "Validation report saved to: %s",
        output_path,
    )


def validate_data(
    df: pd.DataFrame,
) -> None:
    """Execute all validation checks."""

    check_required_columns(df)

    validate_schema(df)

    validate_timestamp_consistency(df)


def main() -> None:
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Validate cleaned water consumption data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    args = parser.parse_args()

    df = load_data(args.input)

    validate_data(df)

    generate_report(
        df,
        args.report,
    )


if __name__ == "__main__":
    main()
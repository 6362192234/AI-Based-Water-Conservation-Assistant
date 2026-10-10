"""
clean.py

Purpose:
    Clean the HSB Living Lab Water Consumption dataset.

Input:
    data/raw/HSB_Living_Lab_Water_Consumption_Anonymized.csv

Output:
    data/processed/cleaned_water_data.parquet
    reports/preprocessing/cleaning_report.csv

Cleaning principles:
    - Remove exact duplicate rows.
    - Preserve legitimate unusual consumption values.
    - Preserve rows with missing contextual information.
    - Handle missing apartment and cluster values explicitly.
    - Detect repeated sensor/timestamp records.
    - Preserve conflicting sensor/timestamp measurements.
    - Flag decreasing cumulative meter values.
    - Do not arbitrarily modify high consumption values.
"""

from pathlib import Path
import argparse
import logging

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT_PATH = Path(
    "data/raw/HSB_Living_Lab_Water_Consumption_Anonymized.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/processed/cleaned_water_data.parquet"
)

DEFAULT_REPORT_PATH = Path(
    "reports/preprocessing/cleaning_report.csv"
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load the raw water consumption dataset.

    Parameters
    ----------
    input_path : Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    logger.info(
        "Loading raw dataset: %s",
        input_path,
    )

    df = pd.read_csv(input_path,sep = ";")

    logger.info(
        "Dataset loaded successfully: %d rows, %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


# ============================================================
# Timestamp Cleaning
# ============================================================

def clean_timestamp(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Convert timestamp to datetime and remove rows
    with invalid timestamps.

    Invalid timestamps cannot be reliably used for
    time-based processing later in the pipeline.
    """

    df = df.copy()

    original_rows = len(df)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_count = int(
        df["timestamp"].isna().sum()
    )

    if invalid_count > 0:
        logger.warning(
            "Removing %d rows with invalid timestamps.",
            invalid_count,
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    logger.info(
        "Timestamp cleaning complete: %d rows removed.",
        original_rows - len(df),
    )

    return df, invalid_count


# ============================================================
# Exact Duplicate Removal
# ============================================================

def remove_exact_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows that are completely identical.

    Only exact duplicates are removed here.

    Repeated sensor/timestamp records with different
    values are handled separately and are NOT deleted.
    """

    before = len(df)

    df = df.drop_duplicates().copy()

    removed_count = before - len(df)

    logger.info(
        "Exact duplicate rows removed: %d",
        removed_count,
    )

    return df, removed_count


# ============================================================
# Missing Context Handling
# ============================================================

def handle_missing_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing contextual values.

    Missing apartment or cluster information does not
    mean that the water-consumption observation itself
    is invalid.

    Therefore, rows are preserved and missing context
    is represented explicitly as UNKNOWN.
    """

    df = df.copy()

    if "apartment" in df.columns:

        apartment_missing = int(
            df["apartment"].isna().sum()
        )

        df["apartment"] = (
            df["apartment"]
            .fillna("UNKNOWN")
        )

        logger.info(
            "Apartment values filled as UNKNOWN: %d",
            apartment_missing,
        )

    if "cluster_name" in df.columns:

        cluster_missing = int(
            df["cluster_name"].isna().sum()
        )

        df["cluster_name"] = (
            df["cluster_name"]
            .fillna("UNKNOWN")
        )

        logger.info(
            "Cluster values filled as UNKNOWN: %d",
            cluster_missing,
        )

    return df


# ============================================================
# Sensor + Timestamp Checks
# ============================================================

def add_sensor_timestamp_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify repeated sensor/timestamp observations.

    Two situations can occur:

    1. Same sensor + timestamp + same value
       -> potentially redundant observation.

    2. Same sensor + timestamp + different values
       -> conflicting measurement.

    We preserve these observations and create flags
    instead of arbitrarily deleting or averaging them.
    """

    df = df.copy()

    duplicate_mask = df.duplicated(
        subset=[
            "sensor_id",
            "timestamp",
        ],
        keep=False,
    )

    df["sensor_timestamp_duplicate"] = (
        duplicate_mask
    )

    different_value_mask = (
        df.groupby(
            [
                "sensor_id",
                "timestamp",
            ]
        )["value"]
        .transform("nunique")
        > 1
    )

    df["sensor_timestamp_value_conflict"] = (
        different_value_mask
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    conflict_rows = int(
        different_value_mask.sum()
    )

    logger.info(
        "Rows involved in repeated sensor/timestamp records: %d",
        duplicate_rows,
    )

    logger.info(
        "Rows involved in conflicting sensor/timestamp values: %d",
        conflict_rows,
    )

    return df


# ============================================================
# Cumulative Meter Checks
# ============================================================

def add_meter_decrease_flag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect decreases in aggregated_value for each sensor.

    A decrease may indicate a data-quality issue, meter
    reset, or another phenomenon.

    These rows are FLAGGED, not deleted.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "sensor_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    previous_value = (
        df.groupby("sensor_id")[
            "aggregated_value"
        ].shift(1)
    )

    df["aggregated_value_decreased"] = (
        df["aggregated_value"]
        < previous_value
    )

    decrease_count = int(
        df[
            "aggregated_value_decreased"
        ].sum()
    )

    logger.info(
        "Aggregated-value decreases flagged: %d",
        decrease_count,
    )

    return df


# ============================================================
# Final Ordering
# ============================================================

def sort_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort cleaned data chronologically.
    """

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# Cleaning Pipeline
# ============================================================

def clean_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the complete cleaning pipeline.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned dataset.

    cleaning_summary : dict
        Summary of cleaning operations.
    """

    original_rows = len(df)

    # --------------------------------------------------------
    # 1. Timestamp cleaning
    # --------------------------------------------------------

    df, invalid_timestamps = clean_timestamp(df)

    # --------------------------------------------------------
    # 2. Remove exact duplicates
    # --------------------------------------------------------

    df, duplicates_removed = (
        remove_exact_duplicates(df)
    )

    # --------------------------------------------------------
    # 3. Handle missing contextual values
    # --------------------------------------------------------

    df = handle_missing_context(df)

    # --------------------------------------------------------
    # 4. Flag sensor/timestamp issues
    # --------------------------------------------------------

    df = add_sensor_timestamp_flags(df)

    # --------------------------------------------------------
    # 5. Flag cumulative meter decreases
    # --------------------------------------------------------

    df = add_meter_decrease_flag(df)

    # --------------------------------------------------------
    # 6. Sort chronologically
    # --------------------------------------------------------

    df = sort_data(df)

    # --------------------------------------------------------
    # Cleaning summary
    # --------------------------------------------------------

    cleaning_summary = {
        "original_rows": original_rows,
        "final_rows": len(df),
        "invalid_timestamps_removed": invalid_timestamps,
        "exact_duplicates_removed": duplicates_removed,
        "rows_removed_total": original_rows - len(df),
        "sensor_timestamp_duplicate_rows": int(
            df["sensor_timestamp_duplicate"].sum()
        ),
        "sensor_timestamp_conflict_rows": int(
            df[
                "sensor_timestamp_value_conflict"
            ].sum()
        ),
        "aggregated_value_decrease_rows": int(
            df[
                "aggregated_value_decreased"
            ].sum()
        ),
    }

    logger.info(
        "Cleaning completed: %d → %d rows",
        original_rows,
        len(df),
    )

    return df, cleaning_summary


# ============================================================
# Save Cleaned Data
# ============================================================

def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save cleaned data as Parquet.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Cleaned dataset saved to: %s",
        output_path,
    )


# ============================================================
# Save Cleaning Report
# ============================================================

def save_cleaning_report(
    summary: dict,
    report_path: Path,
) -> None:
    """
    Save cleaning statistics as CSV.
    """

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(
        {
            "metric": list(
                summary.keys()
            ),
            "value": list(
                summary.values()
            ),
        }
    )

    report.to_csv(
        report_path,
        index=False,
    )

    logger.info(
        "Cleaning report saved to: %s",
        report_path,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Run the complete data-cleaning pipeline.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Clean the HSB Living Lab "
            "water consumption dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw CSV dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the cleaned Parquet dataset.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the cleaning report.",
    )

    args = parser.parse_args()

    # Load
    df = load_data(
        args.input
    )

    # Clean
    cleaned_df, summary = clean_data(
        df
    )

    # Save cleaned dataset
    save_cleaned_data(
        cleaned_df,
        args.output,
    )

    # Save cleaning report
    save_cleaning_report(
        summary,
        args.report,
    )

    logger.info(
        "Data-cleaning pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
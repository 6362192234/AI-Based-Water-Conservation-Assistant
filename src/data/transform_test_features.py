"""
transform_test_features.py

Purpose:
    Transform the test dataset into data/processed/features/test.parquet
    using the exact same feature-engineering logic as the training data
    (as implemented in src/data/feature_engineering.py).

Features engineered (10 required by Isolation Forest and Autoencoder):
    1. value (in m3)
    2. meter_delta
    3. previous_value
    4. rolling_mean
    5. rolling_std
    6. hour_sin
    7. hour_cos
    8. day_of_week_sin
    9. day_of_week_cos
    10. is_weekend

Maintains sensor-wise chronological order (sensor_id, timestamp).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.feature_engineering import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("transform_test_features")

DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "features" / "test.parquet.backup"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "features" / "test.parquet"

REQUIRED_FEATURES = [
    "value",
    "meter_delta",
    "previous_value",
    "rolling_mean",
    "rolling_std",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
]


def transform_test_data(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load test dataset, prepare canonical value and aggregated_value columns,
    apply project's feature-engineering pipeline, sort sensor-wise chronologically,
    and save to output_path.
    """
    if not input_path.exists():
        # Fallback to output_path if backup does not exist
        if output_path.exists():
            input_path = output_path
        else:
            raise FileNotFoundError(
                f"Source test dataset not found at {input_path} or {output_path}."
            )

    logger.info("Loading source test dataset from %s ...", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows with columns: %s", len(df), list(df.columns))

    df = df.copy()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Map measurement columns to project's canonical names expected by build_features()
    # The models were trained on value in m3 (range 0.001 - 0.300 m3)
    if "value" not in df.columns:
        if "value_m3" in df.columns:
            logger.info("Mapping 'value_m3' -> 'value'")
            df["value"] = df["value_m3"]
        elif "value_l" in df.columns:
            logger.info("Mapping 'value_l' / 1000.0 -> 'value'")
            df["value"] = df["value_l"] / 1000.0
        else:
            raise KeyError("Neither 'value', 'value_m3', nor 'value_l' found in test dataset.")

    if "aggregated_value" not in df.columns:
        if "aggregated_m3" in df.columns:
            logger.info("Mapping 'aggregated_m3' -> 'aggregated_value'")
            df["aggregated_value"] = df["aggregated_m3"]
        else:
            raise KeyError("'aggregated_value' or 'aggregated_m3' column is missing from test dataset.")

    # Sort sensor-wise chronologically before engineering features
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    # Apply exact project feature engineering logic
    logger.info("Applying build_features from src.data.feature_engineering...")
    feature_df = build_features(df)

    # Ensure final ordering is sensor-wise chronological
    feature_df = feature_df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    # Verify all 10 features exist
    missing = set(REQUIRED_FEATURES) - set(feature_df.columns)
    if missing:
        raise ValueError(f"Feature engineering failed to produce required columns: {sorted(missing)}")

    logger.info("All 10 required features successfully generated:")
    for f in REQUIRED_FEATURES:
        logger.info("  - %s (dtype: %s, nulls: %d)", f, feature_df[f].dtype, feature_df[f].isna().sum())

    # Save transformed test dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)
    logger.info("Successfully saved transformed test dataset to %s (%d rows)", output_path, len(feature_df))

    return feature_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform test dataset with project feature engineering pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to raw/intermediate test dataset (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save engineered test parquet (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    try:
        transform_test_data(args.input, args.output)
        logger.info("Test feature transformation completed successfully.")
    except Exception as exc:
        logger.error("Failed to transform test features: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

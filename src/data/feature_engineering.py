"""
feature_engineering.py

Purpose:
    Create model-ready features from cleaned water consumption data.

Input:
    Cleaned DataFrame.

Output:
    DataFrame containing engineered features.

Important:
    Feature engineering must avoid future information leakage.
"""

import logging

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------

def add_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create temporal features."""

    df = df.copy()

    timestamp = pd.to_datetime(
        df["timestamp"]
    )

    df["hour"] = timestamp.dt.hour

    df["day_of_week"] = timestamp.dt.dayofweek

    df["month"] = timestamp.dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def add_cyclical_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Represent cyclical time variables mathematically."""

    df = df.copy()

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def add_consumption_change(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate change in cumulative consumption
    for each sensor.
    """

    df = df.copy()

    df = df.sort_values(
        ["sensor_id", "timestamp"]
    )

    df["meter_delta"] = (
        df.groupby("sensor_id")["aggregated_value"]
        .diff()
    )

    return df


def add_consumption_lag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create previous-observation consumption."""

    df = df.copy()

    df = df.sort_values(
        ["sensor_id", "timestamp"]
    )

    df["previous_value"] = (
        df.groupby("sensor_id")["value"]
        .shift(1)
    )

    return df


def add_rolling_features(
    df: pd.DataFrame,
    window: int = 24,
) -> pd.DataFrame:
    """
    Create historical rolling statistics.

    The current observation is excluded using shift(1)
    to avoid using the current value to define its own baseline.
    """

    df = df.copy()

    df = df.sort_values(
        ["sensor_id", "timestamp"]
    )

    historical_value = (
        df.groupby("sensor_id")["value"]
        .shift(1)
    )

    df["rolling_mean"] = (
        historical_value
        .groupby(df["sensor_id"])
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
    )

    df["rolling_std"] = (
        historical_value
        .groupby(df["sensor_id"])
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=2,
            ).std()
        )
    )

    return df


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Execute the complete feature-engineering pipeline."""

    logger.info(
        "Starting feature engineering."
    )

    df = add_temporal_features(df)

    df = add_cyclical_time_features(df)

    df = add_consumption_change(df)

    df = add_consumption_lag(df)

    df = add_rolling_features(df)

    logger.info(
        "Feature engineering completed."
    )

    return df
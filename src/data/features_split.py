"""
features_split.py

Purpose:
    Build model features and perform the project's
    leakage-safe time-based train/validation/test split.

Input:
    Cleaned Parquet dataset.

Output:
    Training, validation, and test datasets.
"""

from pathlib import Path
import argparse
import logging

import pandas as pd

from feature_engineering import build_features


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path(
    "data/processed/cleaned_water_data.parquet"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/processed/features"
)

TRAIN_END = pd.Timestamp(
    "2022-01-01"
)

VALIDATION_END = pd.Timestamp(
    "2022-10-01"
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
# Functions
# -------------------------------------------------------------------

def load_data(
    input_path: Path,
) -> pd.DataFrame:
    """Load cleaned data."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    return pd.read_parquet(input_path)


def split_by_time(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform the project's chronological split.

    Training:
        before 2022-01-01

    Validation:
        2022-01-01 through before 2022-10-01

    Test:
        2022-10-01 onward
    """

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    train = df[
        df["timestamp"] < TRAIN_END
    ].copy()

    validation = df[
        (df["timestamp"] >= TRAIN_END)
        & (df["timestamp"] < VALIDATION_END)
    ].copy()

    test = df[
        df["timestamp"] >= VALIDATION_END
    ].copy()

    logger.info(
        "Training rows: %s",
        len(train),
    )

    logger.info(
        "Validation rows: %s",
        len(validation),
    )

    logger.info(
        "Test rows: %s",
        len(test),
    )

    return train, validation, test


def save_split(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a dataset split as Parquet."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Saved: %s",
        output_path,
    )


def main() -> None:
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description=(
            "Build features and create "
            "time-based train/validation/test splits."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    df = load_data(
        args.input
    )

    feature_df = build_features(
        df
    )

    train, validation, test = split_by_time(
        feature_df
    )

    save_split(
        train,
        args.output_dir / "train.parquet",
    )

    save_split(
        validation,
        args.output_dir / "validation.parquet",
    )

    save_split(
        test,
        args.output_dir / "test.parquet",
    )


if __name__ == "__main__":
    main()
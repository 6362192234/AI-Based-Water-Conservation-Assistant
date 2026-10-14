"""
profile.py

Purpose:
    Profile the raw HSB Living Lab water consumption dataset.

Input:
    data/raw/HSB_Living_Lab_Water_Consumption_Anonymized.csv

Output:
    reports/profiling/raw_profile.html

This module does not clean or modify the raw dataset.
"""

from pathlib import Path
import argparse
import logging

import pandas as pd
from data_profiling import ProfileReport


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path(
    "data/raw/HSB_Living_Lab_Water_Consumption_Anonymized.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "reports/profiling/raw_profile.html"
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

def load_data(input_path: Path) -> pd.DataFrame:
    """Load the raw CSV dataset."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    logger.info("Loading raw dataset: %s", input_path)

    df = pd.read_csv(input_path,sep = ";")

    logger.info(
        "Dataset loaded successfully: %s rows, %s columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def generate_profile(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate and save the raw-data profiling report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Generating raw data profiling report...")

    profile = ProfileReport(
        df,
        title="HSB Water Consumption - Raw Data Profile",
        minimal=True,
        progress_bar=False,
    )

    profile.to_file(output_path)

    logger.info(
        "Raw profiling report saved to: %s",
        output_path,
    )


def main() -> None:
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Profile the raw water consumption dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to raw CSV dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for profiling HTML report.",
    )

    args = parser.parse_args()

    df = load_data(args.input)

    generate_profile(
        df=df,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
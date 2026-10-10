
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "data" / "processed" / "features"
RESULTS_FILE = (
    ROOT / "reports" / "phase4"
    / "isolation_forest_validation.parquet"
)

FEATURES = [
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

validation = pd.read_parquet(FEATURE_DIR / "validation.parquet")
results = pd.read_parquet(RESULTS_FILE)

# Align predictions with validation rows using their original row order.
validation = validation.reset_index(drop=True)
results = results.reset_index(drop=True)

if len(validation) != len(results):
    raise ValueError("Validation data and prediction rows do not match.")

data = validation[["sensor_id", "timestamp"] + FEATURES].copy()
data["anomaly_score"] = results["anomaly_score"]
data["is_anomaly"] = results["is_anomaly"]

for sensor in ["S0142", "S0212", "S0061"]:
    sensor_data = data[data["sensor_id"] == sensor]

    print(f"\n--- {sensor}: Feature comparison ---")
    print("Total readings:", len(sensor_data))
    print("Anomalies:", int(sensor_data["is_anomaly"].sum()))

    # Compare feature distributions between flagged and unflagged readings.
    print("\nMedian feature values:")
    print(
        sensor_data.groupby("is_anomaly")[FEATURES]
        .median()
        .T
        .to_string()
    )

    print("\nMost unusual readings:")
    print(
        sensor_data.nsmallest(10, "anomaly_score")[
            ["timestamp", "value", "meter_delta",
             "previous_value", "rolling_mean",
             "rolling_std", "anomaly_score"]
        ].to_string(index=False)
    )

print("\nFeature investigation completed.")

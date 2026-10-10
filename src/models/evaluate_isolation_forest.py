
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT / "reports" / "phase4"
    / "isolation_forest_validation.parquet"
)

results = pd.read_parquet(RESULTS_FILE)

print("\n--- Validation Anomaly Analysis ---")
print("Total validation readings:", len(results))
print("Anomalies detected:", int(results["is_anomaly"].sum()))
print(
    "Anomaly percentage:",
    round(results["is_anomaly"].mean() * 100, 2),
    "%"
)

print("\n--- Lowest anomaly scores ---")
print(
    results.nsmallest(15, "anomaly_score").to_string(index=False)
)

if "sensor_id" in results.columns:
    print("\n--- Sensors with the most flagged readings ---")
    print(
        results[results["is_anomaly"]]
        .groupby("sensor_id")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .to_string()
    )

if "timestamp" in results.columns:
    print("\n--- Anomalies by month ---")
    results["timestamp"] = pd.to_datetime(results["timestamp"])
    print(
        results[results["is_anomaly"]]
        .groupby(results["timestamp"].dt.to_period("M"))
        .size()
        .to_string()
    )

print("\nAnalysis completed.")


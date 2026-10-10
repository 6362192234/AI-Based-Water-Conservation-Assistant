
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT / "reports" / "phase4"
    / "isolation_forest_validation.parquet"
)

results = pd.read_parquet(RESULTS_FILE)

# Compare each sensor's total readings with its flagged readings.
summary = results.groupby("sensor_id").agg(
    total_readings=("is_anomaly", "size"),
    anomaly_count=("is_anomaly", "sum"),
)

summary["anomaly_rate_pct"] = (
    summary["anomaly_count"] / summary["total_readings"] * 100
)

summary = summary.sort_values(
    "anomaly_rate_pct", ascending=False
)

print("\n--- Sensors with the Highest Anomaly Rates ---")
print(summary.head(15).round(2).to_string())

# Inspect readings around the strongest anomaly sequence.
if "timestamp" in results.columns:
    results["timestamp"] = pd.to_datetime(results["timestamp"])

    sensor = "S0212"
    start = pd.Timestamp("2022-04-17 00:00:00")
    end = pd.Timestamp("2022-04-17 05:00:00")

    period = results[
        (results["sensor_id"] == sensor)
        & (results["timestamp"] >= start)
        & (results["timestamp"] <= end)
    ].sort_values("timestamp")

    print(f"\n--- Readings for {sensor} around the flagged period ---")
    print(period.to_string(index=False))

print("\nInspection completed.")

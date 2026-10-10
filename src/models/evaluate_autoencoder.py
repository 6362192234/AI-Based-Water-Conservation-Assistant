
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT / "reports" / "phase4"
    / "autoencoder_validation.parquet"
)

results = pd.read_parquet(RESULTS_FILE)

print("\n--- Autoencoder Validation Analysis ---")
print("Total readings:", len(results))
print("Anomalies detected:", int(results["is_anomaly"].sum()))
print(
    "Anomaly percentage:",
    round(results["is_anomaly"].mean() * 100, 2),
    "%",
)

print("\n--- 15 Highest Reconstruction Errors ---")
print(
    results.nlargest(15, "reconstruction_error")
    .to_string(index=False)
)

if "sensor_id" in results.columns:
    print("\n--- Sensors with Most Anomalies ---")
    print(
        results[results["is_anomaly"]]
        .groupby("sensor_id")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .to_string()
    )

if "timestamp" in results.columns:
    results["timestamp"] = pd.to_datetime(results["timestamp"])
    print("\n--- Anomalies by Month ---")
    print(
        results[results["is_anomaly"]]
        .groupby(results["timestamp"].dt.to_period("M"))
        .size()
        .to_string()
    )

print("\nEvaluation completed.")

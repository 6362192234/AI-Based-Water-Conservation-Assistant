
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "reports" / "phase4"

IF_FILE = RESULTS_DIR / "isolation_forest_validation.parquet"
AE_FILE = RESULTS_DIR / "autoencoder_validation.parquet"

# Load predictions from both models
if_results = pd.read_parquet(IF_FILE)
ae_results = pd.read_parquet(AE_FILE)

# Confirm both files contain the same validation readings
KEYS = ["timestamp", "sensor_id"]

if not if_results[KEYS].reset_index(drop=True).equals(
    ae_results[KEYS].reset_index(drop=True)
):
    raise ValueError(
        "The validation rows do not match. Align predictions before comparing."
    )

comparison = if_results[KEYS + ["is_anomaly"]].copy()

comparison = comparison.rename(
    columns={"is_anomaly": "if_anomaly"}
)

comparison["ae_anomaly"] = ae_results["is_anomaly"].to_numpy()
comparison["reconstruction_error"] = (
    ae_results["reconstruction_error"].to_numpy()
)

# Classify model agreement
comparison["agreement"] = "Neither flagged"

comparison.loc[
    comparison["if_anomaly"] & comparison["ae_anomaly"],
    "agreement"
] = "Both flagged"

comparison.loc[
    comparison["if_anomaly"] & ~comparison["ae_anomaly"],
    "agreement"
] = "Isolation Forest only"

comparison.loc[
    ~comparison["if_anomaly"] & comparison["ae_anomaly"],
    "agreement"
] = "Autoencoder only"

print("\n--- Model Comparison ---")
print("Total validation readings:", len(comparison))
print(
    "Isolation Forest anomalies:",
    int(comparison["if_anomaly"].sum())
)
print(
    "Autoencoder anomalies:",
    int(comparison["ae_anomaly"].sum())
)

print("\n--- Agreement Summary ---")
print(comparison["agreement"].value_counts().to_string())

both = comparison["if_anomaly"] & comparison["ae_anomaly"]
print("\nReadings flagged by both:", int(both.sum()))

# Compare detections by month
comparison["timestamp"] = pd.to_datetime(comparison["timestamp"])

monthly = comparison.groupby(
    comparison["timestamp"].dt.to_period("M")
).agg(
    isolation_forest=("if_anomaly", "sum"),
    autoencoder=("ae_anomaly", "sum"),
)

print("\n--- Monthly Anomaly Comparison ---")
print(monthly.to_string())

# Save all comparison results
output_file = RESULTS_DIR / "model_comparison.parquet"
comparison.to_parquet(output_file, index=False)

# Save readings flagged by both models for investigation
both_file = RESULTS_DIR / "common_anomalies.parquet"
comparison[both].to_parquet(both_file, index=False)

print("\nComparison saved to:", output_file)
print("Common anomalies saved to:", both_file)

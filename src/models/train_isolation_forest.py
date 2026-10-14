
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Project paths
ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "data" / "processed" / "features"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "reports" / "phase4"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load training and validation data
train = pd.read_parquet(FEATURE_DIR / "train.parquet")
validation = pd.read_parquet(FEATURE_DIR / "validation.parquet")


# 3. Select the initial 10 features
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

for name, df in [("train", train), ("validation", validation)]:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


# 4. Build the baseline model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("detector", IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )),
])


# 5. Train using training data only
print("Training Isolation Forest...")
model.fit(train[FEATURES])
print("Training completed.")


# 6. Score validation data
# Lower decision scores indicate more anomalous observations.
scores = model.decision_function(validation[FEATURES])
predictions = model.predict(validation[FEATURES])

results = pd.DataFrame(index=validation.index)

# Keep useful context for inspecting detected events.
for col in ["timestamp", "sensor_id", "value"]:
    if col in validation.columns:
        results[col] = validation[col].values

results["anomaly_score"] = scores
results["is_anomaly"] = predictions == -1

# 7. Save validation results
output_file = OUTPUT_DIR / "isolation_forest_validation.parquet"
results.to_parquet(output_file, index=False)

# 8. Save model and feature specification
model_file = MODEL_DIR / "isolation_forest_baseline.joblib"
joblib.dump(model, model_file)

feature_file = MODEL_DIR / "isolation_forest_features.json"
with open(feature_file, "w", encoding="utf-8") as f:
    json.dump(FEATURES, f, indent=4)

# 9. Print baseline summary
anomaly_count = int(results["is_anomaly"].sum())
anomaly_rate = results["is_anomaly"].mean() * 100

print("\n--- Isolation Forest Baseline Results ---")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Features used:", len(FEATURES))
print("Validation anomalies:", anomaly_count)
print(f"Validation anomaly rate: {anomaly_rate:.2f}%")
print("Model saved to:", model_file)
print("Features saved to:", feature_file)
print("Validation scores saved to:", output_file)


from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# 1. Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. Project paths
ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "data" / "processed" / "features"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "reports" / "phase4"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load existing Phase 3 datasets
train = pd.read_parquet(FEATURE_DIR / "train.parquet")
validation = pd.read_parquet(FEATURE_DIR / "validation.parquet")

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

# 4. Fit preprocessing on training data only
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(train[FEATURES])
X_train = scaler.fit_transform(X_train_imputed).astype("float32")

X_validation_imputed = imputer.transform(validation[FEATURES])
X_validation = scaler.transform(X_validation_imputed).astype("float32")

# 5. Build the Autoencoder
input_dim = len(FEATURES)

inputs = keras.Input(shape=(input_dim,), name="input_features")
x = layers.Dense(8, activation="relu")(inputs)
encoded = layers.Dense(4, activation="relu", name="bottleneck")(x)
x = layers.Dense(8, activation="relu")(encoded)
outputs = layers.Dense(input_dim, activation="linear")(x)

autoencoder = keras.Model(inputs, outputs, name="water_autoencoder")

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
)

# 6. Train on training data; use validation only to monitor training
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

print("Training Autoencoder...")
history = autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_validation, X_validation),
    epochs=50,
    batch_size=256,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1,
)

print("Training completed.")

# 7. Calculate reconstruction errors
train_reconstructed = autoencoder.predict(X_train, verbose=0)
train_errors = np.mean(
    np.square(X_train - train_reconstructed), axis=1
)

validation_reconstructed = autoencoder.predict(
    X_validation, verbose=0
)
validation_errors = np.mean(
    np.square(X_validation - validation_reconstructed), axis=1
)

# Provisional threshold: 99th percentile of training reconstruction errors.
# This is a starting point, not a verified optimal threshold.
threshold = float(np.percentile(train_errors, 99))

# 8. Save validation predictions
results = pd.DataFrame()

for col in ["timestamp", "sensor_id", "value"]:
    if col in validation.columns:
        results[col] = validation[col].to_numpy()

results["reconstruction_error"] = validation_errors
results["is_anomaly"] = validation_errors > threshold

results_file = OUTPUT_DIR / "autoencoder_validation.parquet"
results.to_parquet(results_file, index=False)

# 9. Save model, preprocessing and feature configuration
model_file = MODEL_DIR / "water_autoencoder.keras"
autoencoder.save(model_file)

joblib.dump(
    {"imputer": imputer, "scaler": scaler},
    MODEL_DIR / "autoencoder_preprocessing.joblib",
)

with open(
    MODEL_DIR / "autoencoder_features.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(FEATURES, f, indent=4)

with open(
    MODEL_DIR / "autoencoder_threshold.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "threshold": threshold,
            "method": "99th percentile of training reconstruction errors",
        },
        f,
        indent=4,
    )

pd.DataFrame(history.history).to_csv(
    OUTPUT_DIR / "autoencoder_training_history.csv",
    index_label="epoch",
)

# 10. Print results
print("\n--- Autoencoder Baseline Results ---")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Features used:", len(FEATURES))
print("Epochs completed:", len(history.history["loss"]))
print("Final training loss:", history.history["loss"][-1])
print("Final validation loss:", history.history["val_loss"][-1])
print("Provisional threshold:", threshold)
print("Validation anomalies:", int(results["is_anomaly"].sum()))
print(
    f"Validation anomaly rate: {results['is_anomaly'].mean() * 100:.2f}%"
)
print("Model saved to:", model_file)
print("Validation predictions saved to:", results_file)

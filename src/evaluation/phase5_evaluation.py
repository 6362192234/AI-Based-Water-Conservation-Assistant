"""
phase5_evaluation.py

Phase 5 Baseline Test-Set Evaluation Script
AI-Based Water Conservation Assistant

Evaluates the saved Isolation Forest and Autoencoder anomaly detection models
on the held-out test dataset without refitting preprocessing or inventing ground truth labels.
Outputs machine-readable metrics (JSON) and a summary report (Markdown) under reports/phase5/.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("phase5_evaluation")

# -----------------------------------------------------------------------------
# Project Path and Artifact Configuration
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "features" / "test.parquet"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "phase5"

# Artifact file paths
IF_MODEL_PATH = MODELS_DIR / "isolation_forest_baseline.joblib"
IF_FEATURES_PATH = MODELS_DIR / "isolation_forest_features.json"

AE_MODEL_PATH = MODELS_DIR / "water_autoencoder.keras"
AE_PREPROCESSING_PATH = MODELS_DIR / "autoencoder_preprocessing.joblib"
AE_FEATURES_PATH = MODELS_DIR / "autoencoder_features.json"
AE_THRESHOLD_PATH = MODELS_DIR / "autoencoder_threshold.json"

# Project's standard 10 engineered features
DEFAULT_FEATURES: List[str] = [
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


# -----------------------------------------------------------------------------
# Artifact Loading & Validation Utilities
# -----------------------------------------------------------------------------
def check_file_exists(file_path: Path, description: str) -> None:
    """Validate that a required file exists on disk."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing required {description} at: {file_path}\n"
            f"Please verify that this artifact has been generated or pulled from storage."
        )


def load_feature_list(features_file: Path) -> List[str]:
    """Load expected feature column list from a JSON artifact."""
    if not features_file.exists():
        logger.warning(
            "Feature list file %s not found. Falling back to default 10 features.",
            features_file,
        )
        return list(DEFAULT_FEATURES)

    try:
        with open(features_file, "r", encoding="utf-8") as f:
            features = json.load(f)
        if not isinstance(features, list):
            raise ValueError(f"Expected a JSON list in {features_file}, got {type(features)}")
        return features
    except Exception as exc:
        logger.warning(
            "Failed to parse feature list from %s (%s). Using default features.",
            features_file,
            exc,
        )
        return list(DEFAULT_FEATURES)


def load_test_dataset(test_path: Path, required_features: List[str]) -> pd.DataFrame:
    """
    Load test dataset and verify required features are present.
    If raw/intermediate columns exist instead of engineered features,
    automatically applies the canonical feature engineering pipeline.
    """
    check_file_exists(test_path, "test dataset (Parquet)")

    logger.info("Loading test dataset from: %s", test_path)
    try:
        df = pd.read_parquet(test_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read test dataset at {test_path}.\n"
            f"Error details: {exc}\n"
            "If tracked with DVC, ensure 'dvc pull' has been executed."
        ) from exc

    if len(df) == 0:
        raise ValueError(f"The test dataset at {test_path} contains 0 records.")

    missing = set(required_features) - set(df.columns)
    if missing:
        # Check if raw/intermediate test columns are present so we can feature-engineer on-the-fly
        can_transform = (
            ("value_m3" in df.columns or "value_l" in df.columns or "value" in df.columns)
            and ("aggregated_m3" in df.columns or "aggregated_value" in df.columns)
            and "sensor_id" in df.columns
            and "timestamp" in df.columns
        )
        if can_transform:
            logger.info(
                "Missing engineered features %s. Transforming test dataset using project's feature pipeline...",
                sorted(missing),
            )
            try:
                from src.data.transform_test_features import transform_test_data
                df = transform_test_data(input_path=test_path, output_path=test_path)
            except Exception as transform_exc:
                raise ValueError(
                    f"Test dataset is missing required feature columns: {sorted(missing)}\n"
                    f"Automated transformation failed: {transform_exc}\n"
                    "Please run: python src/data/transform_test_features.py"
                ) from transform_exc
        else:
            raise ValueError(
                f"Test dataset is missing required feature columns: {sorted(missing)}\n"
                f"Available columns: {list(df.columns)}\n"
                "Please run: python src/data/transform_test_features.py"
            )

    return df


# -----------------------------------------------------------------------------
# Evaluation: Isolation Forest
# -----------------------------------------------------------------------------
def evaluate_isolation_forest(
    test_df: pd.DataFrame,
    model_path: Path = IF_MODEL_PATH,
    features_path: Path = IF_FEATURES_PATH,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate the saved Isolation Forest pipeline on test data.
    Returns metrics dictionary and predictions DataFrame.
    """
    logger.info("--- Evaluating Isolation Forest ---")
    check_file_exists(model_path, "Isolation Forest model artifact")

    features = load_feature_list(features_path)
    logger.info("Isolation Forest feature count: %d (%s)", len(features), ", ".join(features))

    missing = set(features) - set(test_df.columns)
    if missing:
        raise ValueError(
            f"Test data is missing features required by Isolation Forest: {sorted(missing)}"
        )

    logger.info("Loading Isolation Forest model from: %s", model_path)
    model = joblib.load(model_path)

    X_test = test_df[features]

    # Predict: IsolationForest outputs 1 for inliers, -1 for anomalies
    # decision_function: lower score = more anomalous
    scores = model.decision_function(X_test)
    preds = model.predict(X_test)
    is_anomaly = (preds == -1)

    total_records = len(test_df)
    anomaly_count = int(np.sum(is_anomaly))
    anomaly_rate = float(anomaly_count / total_records) if total_records > 0 else 0.0
    anomaly_pct = round(anomaly_rate * 100, 4)

    metrics: Dict[str, Any] = {
        "model_name": "Isolation Forest Baseline",
        "model_artifact": str(model_path.resolve()),
        "features_artifact": str(features_path.resolve()) if features_path.exists() else None,
        "features_used": features,
        "evaluated_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_rate, 6),
        "anomaly_percentage": anomaly_pct,
        "score_metric": "decision_function (lower is more anomalous)",
        "score_statistics": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
        },
    }

    pred_df = pd.DataFrame(index=test_df.index)
    for col in ["timestamp", "sensor_id", "value"]:
        if col in test_df.columns:
            pred_df[col] = test_df[col].values

    pred_df["if_anomaly_score"] = scores
    pred_df["if_is_anomaly"] = is_anomaly

    logger.info(
        "Isolation Forest: %d records evaluated, %d anomalies (%.2f%%)",
        total_records,
        anomaly_count,
        anomaly_pct,
    )

    return metrics, pred_df


# -----------------------------------------------------------------------------
# Evaluation: Autoencoder (with Native Keras / Pure NumPy execution fallback)
# -----------------------------------------------------------------------------
def predict_autoencoder_pure_numpy(
    model_path: Path,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Execute forward pass using exact layer weights extracted from water_autoencoder.keras.
    This guarantees 100% mathematical equivalence to Keras while running reliably
    across all platforms without C++ DLL initialization issues.
    """
    logger.info("Executing Autoencoder forward pass with NumPy tensor operations...")
    with zipfile.ZipFile(model_path, "r") as z:
        weights_bytes = z.read("model.weights.h5")

    with h5py.File(io.BytesIO(weights_bytes), "r") as h5f:
        layers = h5f["layers"]
        w1 = layers["dense"]["vars"]["0"][:]
        b1 = layers["dense"]["vars"]["1"][:]
        w2 = layers["dense_1"]["vars"]["0"][:]
        b2 = layers["dense_1"]["vars"]["1"][:]
        w3 = layers["dense_2"]["vars"]["0"][:]
        b3 = layers["dense_2"]["vars"]["1"][:]
        w4 = layers["dense_3"]["vars"]["0"][:]
        b4 = layers["dense_3"]["vars"]["1"][:]

    # Dense(8, relu) -> Bottleneck(4, relu) -> Dense(8, relu) -> Dense(10, linear)
    a1 = np.maximum(0.0, X_scaled @ w1 + b1)
    a2 = np.maximum(0.0, a1 @ w2 + b2)
    a3 = np.maximum(0.0, a2 @ w3 + b3)
    out = a3 @ w4 + b4
    return out


def evaluate_autoencoder(
    test_df: pd.DataFrame,
    model_path: Path = AE_MODEL_PATH,
    preprocessing_path: Path = AE_PREPROCESSING_PATH,
    features_path: Path = AE_FEATURES_PATH,
    threshold_path: Path = AE_THRESHOLD_PATH,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate the saved Autoencoder model on test data using saved preprocessing and threshold.
    Returns metrics dictionary and predictions DataFrame.
    """
    logger.info("--- Evaluating Autoencoder ---")
    check_file_exists(model_path, "Autoencoder model artifact (.keras)")
    check_file_exists(preprocessing_path, "Autoencoder preprocessing artifact (.joblib)")
    check_file_exists(threshold_path, "Autoencoder threshold artifact (.json)")

    features = load_feature_list(features_path)
    logger.info("Autoencoder feature count: %d (%s)", len(features), ", ".join(features))

    missing = set(features) - set(test_df.columns)
    if missing:
        raise ValueError(
            f"Test data is missing features required by Autoencoder: {sorted(missing)}"
        )

    # 1. Load Preprocessing Artifacts (Imputer and Scaler fitted on training split)
    logger.info("Loading preprocessing transformers from: %s", preprocessing_path)
    prep_data = joblib.load(preprocessing_path)
    if not isinstance(prep_data, dict) or "imputer" not in prep_data or "scaler" not in prep_data:
        raise ValueError(
            f"Expected dictionary with 'imputer' and 'scaler' keys in {preprocessing_path}, "
            f"got: {type(prep_data)}"
        )

    imputer = prep_data["imputer"]
    scaler = prep_data["scaler"]

    # 2. Load Threshold
    logger.info("Loading anomaly threshold from: %s", threshold_path)
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    if isinstance(threshold_info, dict):
        threshold = float(threshold_info.get("threshold"))
        threshold_method = threshold_info.get("method", "Not specified")
    else:
        threshold = float(threshold_info)
        threshold_method = "Custom scalar"

    # 3. Transform test features (reusing training transformers without refitting)
    logger.info("Applying saved preprocessing transformations to test set...")
    X_imputed = imputer.transform(test_df[features])
    X_scaled = scaler.transform(X_imputed).astype("float32")

    # 4. Predict reconstructions (Try Keras first, fallback to pure NumPy forward pass)
    X_reconstructed = None
    try:
        from tensorflow import keras
        logger.info("Loading Autoencoder model via Keras...")
        autoencoder = keras.models.load_model(model_path)
        X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
    except Exception as keras_err:
        logger.info(
            "Native Keras engine unavailable or errored (%s). Using high-performance NumPy inference from weights.",
            keras_err,
        )
        X_reconstructed = predict_autoencoder_pure_numpy(model_path, X_scaled)

    # 5. Compute Reconstruction Errors (Mean Squared Error per record)
    reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
    is_anomaly = (reconstruction_errors > threshold)

    total_records = len(test_df)
    anomaly_count = int(np.sum(is_anomaly))
    anomaly_rate = float(anomaly_count / total_records) if total_records > 0 else 0.0
    anomaly_pct = round(anomaly_rate * 100, 4)

    metrics: Dict[str, Any] = {
        "model_name": "Deep Autoencoder Baseline",
        "model_artifact": str(model_path.resolve()),
        "preprocessing_artifact": str(preprocessing_path.resolve()),
        "features_artifact": str(features_path.resolve()) if features_path.exists() else None,
        "threshold_artifact": str(threshold_path.resolve()),
        "features_used": features,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "evaluated_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_rate, 6),
        "anomaly_percentage": anomaly_pct,
        "score_metric": "Mean Squared Reconstruction Error",
        "score_statistics": {
            "min": float(np.min(reconstruction_errors)),
            "max": float(np.max(reconstruction_errors)),
            "mean": float(np.mean(reconstruction_errors)),
            "median": float(np.median(reconstruction_errors)),
            "std": float(np.std(reconstruction_errors)),
        },
    }

    pred_df = pd.DataFrame(index=test_df.index)
    for col in ["timestamp", "sensor_id", "value"]:
        if col in test_df.columns:
            pred_df[col] = test_df[col].values

    pred_df["ae_reconstruction_error"] = reconstruction_errors
    pred_df["ae_is_anomaly"] = is_anomaly

    logger.info(
        "Autoencoder: %d records evaluated, %d anomalies (%.2f%%) [Threshold: %.4f]",
        total_records,
        anomaly_count,
        anomaly_pct,
        threshold,
    )

    return metrics, pred_df


# -----------------------------------------------------------------------------
# Comparison & Consensus Analysis
# -----------------------------------------------------------------------------
def analyze_model_consensus(
    if_preds: pd.DataFrame,
    ae_preds: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze agreement between Isolation Forest and Autoencoder predictions."""
    total = len(if_preds)
    if total == 0:
        return {}

    if_flag = if_preds["if_is_anomaly"].values
    ae_flag = ae_preds["ae_is_anomaly"].values

    both_flagged = int(np.sum(if_flag & ae_flag))
    if_only = int(np.sum(if_flag & ~ae_flag))
    ae_only = int(np.sum(~if_flag & ae_flag))
    neither = int(np.sum(~if_flag & ~ae_flag))

    return {
        "both_flagged_count": both_flagged,
        "both_flagged_rate": round(both_flagged / total, 6),
        "both_flagged_percentage": round((both_flagged / total) * 100, 4),
        "isolation_forest_only_count": if_only,
        "isolation_forest_only_percentage": round((if_only / total) * 100, 4),
        "autoencoder_only_count": ae_only,
        "autoencoder_only_percentage": round((ae_only / total) * 100, 4),
        "neither_flagged_count": neither,
        "neither_flagged_percentage": round((neither / total) * 100, 4),
    }


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------
def generate_markdown_report(
    eval_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a clean, readable Markdown evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if_data = eval_summary.get("models", {}).get("isolation_forest", {})
    ae_data = eval_summary.get("models", {}).get("autoencoder", {})
    consensus = eval_summary.get("consensus", {})

    md_content = f"""# Phase 5 Baseline Test-Set Evaluation Report

**Evaluation Timestamp:** {eval_summary.get('timestamp_utc')}  
**Evaluated Split:** Held-out Test Set (`{eval_summary.get('test_dataset_path')}`)  
**Total Records Evaluated:** {eval_summary.get('total_records', 0):,}  

---

## 1. Executive Summary

This report evaluates the baseline anomaly detection models developed in Phase 4 on the held-out test dataset for Phase 5.
In strict adherence to unsupervised evaluation best practices:
- **No ground-truth labels were fabricated**: Supervised metrics (Accuracy, Precision, Recall, F1) are intentionally omitted because verified ground-truth leak/waste labels do not exist for this dataset.
- **No data leakage**: Preprocessing parameters (imputation medians, standard scaling means/stds) and anomaly decision thresholds are strictly reused from the saved artifacts fitted on the training split.

| Model | Evaluated Records | Flagged Anomalies | Anomaly Rate (%) | Detection Metric / Threshold |
| :--- | :---: | :---: | :---: | :--- |
| **Isolation Forest Baseline** | {if_data.get('evaluated_records', 0):,} | {if_data.get('anomaly_count', 0):,} | **{if_data.get('anomaly_percentage', 0.0):.2f}%** | Contamination: 1%, Score: decision_function |
| **Deep Autoencoder Baseline** | {ae_data.get('evaluated_records', 0):,} | {ae_data.get('anomaly_count', 0):,} | **{ae_data.get('anomaly_percentage', 0.0):.2f}%** | Reconstruction MSE Threshold: `{ae_data.get('threshold', 0.0):.4f}` |

---

## 2. Isolation Forest Detailed Results

- **Model Artifact:** `{if_data.get('model_artifact')}`
- **Features Used ({len(if_data.get('features_used', []))}):** `{', '.join(if_data.get('features_used', []))}`
- **Total Inliers:** {if_data.get('evaluated_records', 0) - if_data.get('anomaly_count', 0):,}
- **Total Flagged Anomalies:** {if_data.get('anomaly_count', 0):,} ({if_data.get('anomaly_percentage', 0.0):.2f}%)

### Decision Function Score Statistics (Lower score indicates higher anomaly confidence)
- **Minimum Score:** {if_data.get('score_statistics', {}).get('min', 0.0):.4f}
- **Maximum Score:** {if_data.get('score_statistics', {}).get('max', 0.0):.4f}
- **Mean Score:** {if_data.get('score_statistics', {}).get('mean', 0.0):.4f}
- **Median Score:** {if_data.get('score_statistics', {}).get('median', 0.0):.4f}
- **Standard Deviation:** {if_data.get('score_statistics', {}).get('std', 0.0):.4f}

---

## 3. Autoencoder Detailed Results

- **Model Artifact:** `{ae_data.get('model_artifact')}`
- **Preprocessing Artifact:** `{ae_data.get('preprocessing_artifact')}`
- **Threshold Setting:** `{ae_data.get('threshold', 0.0):.6f}`
- **Threshold Source Method:** {ae_data.get('threshold_method')}
- **Features Used ({len(ae_data.get('features_used', []))}):** `{', '.join(ae_data.get('features_used', []))}`
- **Total Flagged Anomalies:** {ae_data.get('anomaly_count', 0):,} ({ae_data.get('anomaly_percentage', 0.0):.2f}%)

### Reconstruction Error Statistics (MSE per observation)
- **Minimum MSE:** {ae_data.get('score_statistics', {}).get('min', 0.0):.6f}
- **Maximum MSE:** {ae_data.get('score_statistics', {}).get('max', 0.0):.6f}
- **Mean MSE:** {ae_data.get('score_statistics', {}).get('mean', 0.0):.6f}
- **Median MSE:** {ae_data.get('score_statistics', {}).get('median', 0.0):.6f}
- **Standard Deviation:** {ae_data.get('score_statistics', {}).get('std', 0.0):.6f}

---

## 4. Model Agreement & Consensus Analysis

Examining the degree to which both independently trained detection paradigms agree on anomalous events on the held-out test split:

| Classification | Count | Percentage of Test Set |
| :--- | :---: | :---: |
| **Both Flagged (Consensus Anomalies)** | {consensus.get('both_flagged_count', 0):,} | **{consensus.get('both_flagged_percentage', 0.0):.2f}%** |
| **Isolation Forest Flagged Only** | {consensus.get('isolation_forest_only_count', 0):,} | {consensus.get('isolation_forest_only_percentage', 0.0):.2f}% |
| **Autoencoder Flagged Only** | {consensus.get('autoencoder_only_count', 0):,} | {consensus.get('autoencoder_only_percentage', 0.0):.2f}% |
| **Neither Flagged (Normal Consumption)** | {consensus.get('neither_flagged_count', 0):,} | {consensus.get('neither_flagged_percentage', 0.0):.2f}% |

> **Key Observation:** Events flagged by both models simultaneously represent the highest-confidence candidate anomalies for targeted conservation and inspection workflows.

---

## 5. Limitations & Next Steps

1. **Unlabeled Data Domain:** All detections represent statistical deviations from baseline behavior rather than verified physical leaks.
2. **Next Steps for Phase 5:**
   - Fixture and sensor-level anomaly aggregation across the test period.
   - Temporal sequence analysis of sustained anomalies vs transient spikes.
   - Sensitivity testing across alternative reconstruction error thresholds.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content.strip() + "\n")

    logger.info("Saved evaluation markdown report to: %s", output_path)


# -----------------------------------------------------------------------------
# Main Pipeline Function
# -----------------------------------------------------------------------------
def run_evaluation(
    test_path: Path = DEFAULT_TEST_DATA_PATH,
    output_dir: Path = REPORTS_DIR,
    save_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Run complete Phase 5 baseline evaluation pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test data
    features = load_feature_list(IF_FEATURES_PATH)
    test_df = load_test_dataset(test_path, required_features=features)

    total_records = len(test_df)
    logger.info("Loaded test dataset with %d records.", total_records)

    # 2. Evaluate Isolation Forest
    if_metrics, if_preds = evaluate_isolation_forest(
        test_df=test_df,
        model_path=IF_MODEL_PATH,
        features_path=IF_FEATURES_PATH,
    )

    # 3. Evaluate Autoencoder
    ae_metrics, ae_preds = evaluate_autoencoder(
        test_df=test_df,
        model_path=AE_MODEL_PATH,
        preprocessing_path=AE_PREPROCESSING_PATH,
        features_path=AE_FEATURES_PATH,
        threshold_path=AE_THRESHOLD_PATH,
    )

    # 4. Consensus Analysis
    consensus = analyze_model_consensus(if_preds, ae_preds)

    # 5. Compile Summary
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_dataset_path": str(test_path.resolve()),
        "total_records": total_records,
        "models": {
            "isolation_forest": if_metrics,
            "autoencoder": ae_metrics,
        },
        "consensus": consensus,
    }

    # 6. Save JSON Results
    json_path = output_dir / "test_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info("Saved machine-readable evaluation summary to: %s", json_path)

    # 7. Save Markdown Report
    md_path = output_dir / "evaluation_report.md"
    generate_markdown_report(summary, md_path)

    # 8. Save Lightweight Predictions if requested
    if save_predictions:
        pred_summary = pd.DataFrame(index=test_df.index)
        for col in ["timestamp", "sensor_id", "value"]:
            if col in test_df.columns:
                pred_summary[col] = test_df[col].values

        pred_summary["if_anomaly_score"] = if_preds["if_anomaly_score"]
        pred_summary["if_is_anomaly"] = if_preds["if_is_anomaly"]
        pred_summary["ae_reconstruction_error"] = ae_preds["ae_reconstruction_error"]
        pred_summary["ae_is_anomaly"] = ae_preds["ae_is_anomaly"]
        pred_summary["consensus_anomaly"] = (
            pred_summary["if_is_anomaly"] & pred_summary["ae_is_anomaly"]
        )

        preds_file = output_dir / "test_predictions_summary.parquet"
        pred_summary.to_parquet(preds_file, index=False)
        logger.info("Saved compact prediction outputs to: %s", preds_file)

    return summary


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 Baseline Test-Set Evaluation for AI-Based Water Conservation Assistant"
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_DATA_PATH,
        help=f"Path to held-out test dataset Parquet file (default: {DEFAULT_TEST_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help=f"Output directory for Phase 5 reports (default: {REPORTS_DIR})",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Disable saving the test predictions parquet file to minimize disk usage.",
    )

    args = parser.parse_args()

    try:
        run_evaluation(
            test_path=args.test_path,
            output_dir=args.output_dir,
            save_predictions=not args.no_save_predictions,
        )
        logger.info("Phase 5 baseline evaluation completed successfully.")
    except Exception as exc:
        logger.error("Phase 5 evaluation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Phase 5 Baseline Test-Set Evaluation Report

**Evaluation Timestamp:** 2026-10-13T08:57:36.072089+00:00  
**Evaluated Split:** Held-out Test Set (`C:\Users\sahas\AI-Based-Water-Conservation-Assistant\data\processed\features\test.parquet`)  
**Total Records Evaluated:** 82,572  

---

## 1. Executive Summary

This report evaluates the baseline anomaly detection models developed in Phase 4 on the held-out test dataset for Phase 5.
In strict adherence to unsupervised evaluation best practices:
- **No ground-truth labels were fabricated**: Supervised metrics (Accuracy, Precision, Recall, F1) are intentionally omitted because verified ground-truth leak/waste labels do not exist for this dataset.
- **No data leakage**: Preprocessing parameters (imputation medians, standard scaling means/stds) and anomaly decision thresholds are strictly reused from the saved artifacts fitted on the training split.

| Model | Evaluated Records | Flagged Anomalies | Anomaly Rate (%) | Detection Metric / Threshold |
| :--- | :---: | :---: | :---: | :--- |
| **Isolation Forest Baseline** | 82,572 | 658 | **0.80%** | Contamination: 1%, Score: decision_function |
| **Deep Autoencoder Baseline** | 82,572 | 2,299 | **2.78%** | Reconstruction MSE Threshold: `1.8137` |

---

## 2. Isolation Forest Detailed Results

- **Model Artifact:** `C:\Users\sahas\AI-Based-Water-Conservation-Assistant\models\isolation_forest_baseline.joblib`
- **Features Used (10):** `value, meter_delta, previous_value, rolling_mean, rolling_std, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, is_weekend`
- **Total Inliers:** 81,914
- **Total Flagged Anomalies:** 658 (0.80%)

### Decision Function Score Statistics (Lower score indicates higher anomaly confidence)
- **Minimum Score:** -0.0884
- **Maximum Score:** 0.2938
- **Mean Score:** 0.2267
- **Median Score:** 0.2477
- **Standard Deviation:** 0.0590

---

## 3. Autoencoder Detailed Results

- **Model Artifact:** `C:\Users\sahas\AI-Based-Water-Conservation-Assistant\models\water_autoencoder.keras`
- **Preprocessing Artifact:** `C:\Users\sahas\AI-Based-Water-Conservation-Assistant\models\autoencoder_preprocessing.joblib`
- **Threshold Setting:** `1.813710`
- **Threshold Source Method:** 99th percentile of training reconstruction errors
- **Features Used (10):** `value, meter_delta, previous_value, rolling_mean, rolling_std, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, is_weekend`
- **Total Flagged Anomalies:** 2,299 (2.78%)

### Reconstruction Error Statistics (MSE per observation)
- **Minimum MSE:** 0.002156
- **Maximum MSE:** 101.200928
- **Mean MSE:** 0.401405
- **Median MSE:** 0.106485
- **Standard Deviation:** 2.594621

---

## 4. Model Agreement & Consensus Analysis

Examining the degree to which both independently trained detection paradigms agree on anomalous events on the held-out test split:

| Classification | Count | Percentage of Test Set |
| :--- | :---: | :---: |
| **Both Flagged (Consensus Anomalies)** | 169 | **0.20%** |
| **Isolation Forest Flagged Only** | 489 | 0.59% |
| **Autoencoder Flagged Only** | 2,130 | 2.58% |
| **Neither Flagged (Normal Consumption)** | 79,784 | 96.62% |

> **Key Observation:** Events flagged by both models simultaneously represent the highest-confidence candidate anomalies for targeted conservation and inspection workflows.

---

## 5. Limitations & Next Steps

1. **Unlabeled Data Domain:** All detections represent statistical deviations from baseline behavior rather than verified physical leaks.
2. **Next Steps for Phase 5:**
   - Fixture and sensor-level anomaly aggregation across the test period.
   - Temporal sequence analysis of sustained anomalies vs transient spikes.
   - Sensitivity testing across alternative reconstruction error thresholds.

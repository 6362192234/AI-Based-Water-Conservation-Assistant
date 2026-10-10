# Isolation Forest Baseline Evaluation

## Dataset
- Training observations: 451,670
- Validation observations: 117,933
- Selected features: 10

## Results
- Validation anomalies detected: 1,011
- Validation anomaly rate: 0.86%
- Model: Isolation Forest
- Number of trees: 100
- Configured contamination: 1%

## Key observations
- Anomalies were concentrated in sensors S0142, S0212, and S0061.
- S0142 had 260 flagged readings, S0212 had 369, and S0061 had 161.
- Flagged readings for these sensors generally had higher consumption values than their unflagged readings.
- S0212 showed a prolonged sequence of flagged readings in April 2022.

## Limitations
These detections are candidate anomalies, not confirmed leaks or water wastage. Reliable ground-truth anomaly labels are not available, so detection accuracy has not yet been established. Differences in normal consumption between sensors may influence the predictions.

## Conclusion
The Isolation Forest baseline trained successfully and generated validation anomaly scores. Further evaluation and comparison with an Autoencoder are required before selecting the final model.
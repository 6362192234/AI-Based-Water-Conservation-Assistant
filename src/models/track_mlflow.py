
from pathlib import Path

import mlflow
import pandas as pd


# --------------------------------------------------
# 1. Project paths and MLflow configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB = ROOT / "mlflow.db"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "phase4"

# Use SQLite instead of the deprecated filesystem backend
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

# Create or select the experiment
mlflow.set_experiment("Water-Conservation-Anomaly-Detection")


# --------------------------------------------------
# 2. Function to track a trained model
# --------------------------------------------------

def log_existing_model(
    run_name,
    model_file,
    results_file,
    model_type,
    params,
    score_column,
    extra_artifacts=None,
    history_file=None,
):
    """Log an existing model's settings, results and artifacts."""

    if not results_file.exists():
        raise FileNotFoundError(
            f"Validation results not found: {results_file}"
        )

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}"
        )

    results = pd.read_parquet(results_file)

    if "is_anomaly" not in results.columns:
        raise ValueError(
            f"'is_anomaly' column missing from {results_file}"
        )

    # Handle Boolean or integer anomaly indicators
    anomaly_flags = results["is_anomaly"].astype(bool)

    anomaly_count = int(anomaly_flags.sum())
    anomaly_rate = float(anomaly_flags.mean())

    with mlflow.start_run(run_name=run_name):

        # Log model configuration
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("feature_count", 10)

        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log validation metrics
        mlflow.log_metric("validation_rows", len(results))
        mlflow.log_metric(
            "validation_anomalies", anomaly_count
        )
        mlflow.log_metric(
            "validation_anomaly_rate", anomaly_rate
        )

        # Log mean score, if available
        if score_column in results.columns:
            mlflow.log_metric(
                "mean_validation_score",
                float(results[score_column].mean()),
            )

        # Save model and validation results as artifacts
        mlflow.log_artifact(
            str(model_file),
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(results_file),
            artifact_path="validation",
        )

        # Log optional training history
        if history_file and history_file.exists():
            mlflow.log_artifact(
                str(history_file),
                artifact_path="training",
            )

        # Log additional supporting files
        for artifact in extra_artifacts or []:
            if artifact.exists():
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path="supporting_files",
                )
            else:
                print(f"Warning: Artifact not found: {artifact}")

        print(f"\nLogged experiment: {run_name}")
        print(f"Model type: {model_type}")
        print(f"Validation rows: {len(results):,}")
        print(f"Anomalies: {anomaly_count:,}")
        print(f"Anomaly rate: {anomaly_rate:.2%}")


# --------------------------------------------------
# 3. Track Isolation Forest
# --------------------------------------------------

def track_isolation_forest():

    log_existing_model(
        run_name="Isolation-Forest-Baseline",

        model_file=(
            MODELS_DIR / "isolation_forest_baseline.joblib"
        ),

        results_file=(
            REPORTS_DIR / "isolation_forest_validation.parquet"
        ),

        model_type="IsolationForest",

        params={
            "n_estimators": 100,
            "contamination": 0.01,
            "random_state": 42,
            "n_jobs": -1,
        },

        score_column="anomaly_score",

        extra_artifacts=[
            MODELS_DIR / "isolation_forest_features.json",
        ],
    )


# --------------------------------------------------
# 4. Track Autoencoder
# --------------------------------------------------

def track_autoencoder():

    threshold_file = (
        MODELS_DIR / "autoencoder_threshold.json"
    )

    threshold = None

    if threshold_file.exists():
        import json

        with open(threshold_file, "r", encoding="utf-8") as file:
            threshold_data = json.load(file)

        # Support the threshold file's stored value
        if isinstance(threshold_data, dict):
            threshold = threshold_data.get("threshold")
        else:
            threshold = threshold_data

    params = {
        "batch_size": 256,
        "learning_rate": 0.001,
        "threshold_method": "training_error_99th_percentile",
    }

    if threshold is not None:
        params["reconstruction_error_threshold"] = float(
            threshold
        )

    log_existing_model(
        run_name="Autoencoder-Baseline",

        model_file=(
            MODELS_DIR / "water_autoencoder.keras"
        ),

        results_file=(
            REPORTS_DIR / "autoencoder_validation.parquet"
        ),

        model_type="Autoencoder",

        params=params,

        score_column="reconstruction_error",

        history_file=(
            REPORTS_DIR / "autoencoder_training_history.csv"
        ),

        extra_artifacts=[
            MODELS_DIR / "autoencoder_preprocessing.joblib",
            MODELS_DIR / "autoencoder_features.json",
            threshold_file,
        ],
    )


# --------------------------------------------------
# 5. Track model comparison
# --------------------------------------------------

def track_model_comparison():

    comparison_file = (
        REPORTS_DIR / "model_comparison.parquet"
    )

    if not comparison_file.exists():
        print(
            "\nWarning: Model comparison file not found. "
            "Skipping comparison run."
        )
        return

    comparison = pd.read_parquet(comparison_file)

    with mlflow.start_run(
        run_name="Model-Comparison"
    ):

        mlflow.log_param(
            "models_compared",
            "IsolationForest,Autoencoder",
        )

        mlflow.log_metric(
            "comparison_rows",
            len(comparison),
        )

        # Log anomaly counts when comparison columns exist
        for column in [
            "isolation_forest_anomaly",
            "autoencoder_anomaly",
        ]:
            if column in comparison.columns:
                mlflow.log_metric(
                    f"{column}_count",
                    int(comparison[column].astype(bool).sum()),
                )

        mlflow.log_artifact(
            str(comparison_file),
            artifact_path="comparison",
        )

        common_file = (
            REPORTS_DIR / "common_anomalies.parquet"
        )

        if common_file.exists():
            common = pd.read_parquet(common_file)

            mlflow.log_metric(
                "common_anomaly_count",
                len(common),
            )

            mlflow.log_artifact(
                str(common_file),
                artifact_path="comparison",
            )

        print("\nLogged experiment: Model-Comparison")
        print(f"Comparison rows: {len(comparison):,}")


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("MLflow Experiment Tracking")
    print("=" * 55)

    print(f"Project directory: {ROOT}")
    print(f"MLflow database: {MLFLOW_DB}")

    track_isolation_forest()
    track_autoencoder()
    track_model_comparison()

    print("\nAll available experiments have been logged.")
    print("Start the MLflow UI to inspect the results.")


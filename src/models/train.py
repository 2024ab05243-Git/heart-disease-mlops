import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

# --- PATH LOGIC (Environment Aware) ---
# This ensures we never use 'C:' on a Linux GitHub Runner
if os.getenv("GITHUB_WORKSPACE"):
    # Running on GitHub Actions (Linux)
    BASE_DIR = Path(os.getenv("GITHUB_WORKSPACE"))
else:
    # Running locally (Windows/Mac)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

MLRUNS_DIR = BASE_DIR / "mlruns"
DATA_PATH = BASE_DIR / "data" / "processed" / "heart_disease_processed.csv"

# Ensure directories exist
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Set Tracking URI using proper file scheme
# .as_posix() converts Windows \ to Linux /
# tracking_uri = f"file://{MLRUNS_DIR.as_posix()}"
tracking_uri = MLRUNS_DIR.resolve().as_uri()

mlflow.set_tracking_uri(tracking_uri)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Check your pathing.")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_and_evaluate(model, X, y):
    """
    Core logic for Task 2: Model Development
    Uses 5-fold Stratified Cross-Validation
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    return {metric: np.mean(results[f"test_{metric}"]) for metric in scoring.keys()}


def main():
    print("Starting MLflow training session...")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    X, y = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        # Start MLflow run (Task 3: Experiment Tracking)
        mlflow.set_experiment("heart-disease-ci")
        with mlflow.start_run(run_name=name):
            # Fit model on the full set to create the final artifact
            model.fit(X, y)

            # Log hyperparameters
            mlflow.log_param("model_type", name)
            if name == "RandomForest":
                mlflow.log_param("n_estimators", 200)

            # Log metrics (Task 3)
            metrics = train_and_evaluate(model, X, y)
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Log Model Artifact with Signature (Task 4: Packaging)
            # Signature allows the API to know exactly what input columns to expect
            signature = infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X.iloc[:3],
            )

            print(f"Successfully logged {name} to MLflow.")


if __name__ == "__main__":
    main()

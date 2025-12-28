import os
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

# FIX: Dynamically define absolute paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLRUNS_DIR = BASE_DIR / "mlruns"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "heart_disease_processed.csv"

# Ensure the mlruns directory exists
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Set tracking URI using an absolute path (file://...)
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR.as_posix()}")

def load_data():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run EDA/preprocessing first.")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def train_and_evaluate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    return {
        metric: results[f"test_{metric}"].mean()
        for metric in scoring.keys()
    }

def main():
    X, y = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        # Start MLflow run 
        with mlflow.start_run(run_name=name):
            # Fit model on full data for logging
            model.fit(X, y) 
            
            # Log parameters 
            mlflow.log_param("model_type", name)
            
            # Evaluate using cross-validation [cite: 20]
            metrics = train_and_evaluate(model, X, y)
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Task 4: Log model with signature for reproducibility [cite: 29, 30]
            signature = infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model",
                signature=signature,
                input_example=X.iloc[:3]
            )

            print(f"\n{name} Performance logged to MLflow.")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
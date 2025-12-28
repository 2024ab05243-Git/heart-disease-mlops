import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


PROCESSED_DATA_PATH = Path("data/processed/heart_disease_processed.csv")


def load_data():
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
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)

            metrics = train_and_evaluate(model, X, y)

            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"\n{name} Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    

if __name__ == "__main__":
    main()

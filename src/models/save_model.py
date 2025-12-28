import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.features.build_features import preprocess_data

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
PIPELINE_PATH = MODEL_DIR / "preprocessing_pipeline.joblib"


def train_and_save():
    processed_df, pipeline = preprocess_data()

    X = processed_df.drop("target", axis=1)
    y = processed_df["target"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)

    print("Model and preprocessing pipeline saved successfully.")


if __name__ == "__main__":
    train_and_save()

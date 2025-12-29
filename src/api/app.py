from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

model = joblib.load(MODEL_DIR / "random_forest_model.joblib")
pipeline = joblib.load(MODEL_DIR / "preprocessing_pipeline.joblib")

app = FastAPI(title="Heart Disease Prediction API")


class PatientInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.post("/predict")
def predict(data: PatientInput):
    X = np.array([list(data.dict().values())])
    X_processed = pipeline.transform(X)

    prediction = int(model.predict(X_processed)[0])
    confidence = float(max(model.predict_proba(X_processed)[0]))

    return {
        "prediction": prediction,
        "confidence": confidence
    }

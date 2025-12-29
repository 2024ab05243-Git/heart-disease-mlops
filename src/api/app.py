import logging
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -------------------------------------------------------------------
# Load model and preprocessing pipeline
# -------------------------------------------------------------------
MODEL_DIR = Path("models")

model = joblib.load(MODEL_DIR / "random_forest_model.joblib")
pipeline = joblib.load(MODEL_DIR / "preprocessing_pipeline.joblib")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")

# -------------------------------------------------------------------
# Prometheus metrics
# -------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint"],
)


# -------------------------------------------------------------------
# Middleware: logging + metrics
# -------------------------------------------------------------------
@app.middleware("http")
async def logging_and_metrics(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
    ).inc()

    logging.info(
        f"Completed request: {request.method} {request.url.path} "
        f"Status: {response.status_code}"
    )
    return response


# -------------------------------------------------------------------
# Input schema
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------
@app.post("/predict")
def predict(data: PatientInput):
    X = np.array([list(data.dict().values())])
    X_processed = pipeline.transform(X)

    prediction = int(model.predict(X_processed)[0])
    confidence = float(max(model.predict_proba(X_processed)[0]))

    logging.info(f"Prediction made: prediction={prediction}, confidence={confidence}")

    return {
        "prediction": prediction,
        "confidence": confidence,
    }


# -------------------------------------------------------------------
# Metrics endpoint
# -------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain",
    )

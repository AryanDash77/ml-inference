"""
ML Inference API - Fraud Detection

Loads the real fraud_xgb_model.pkl (30 features: V1-V28, scaled_amount,
scaled_time - same order the model was trained on) plus two SEPARATE
scalers (fixing the original single-scaler bug where fit_transform on
Time silently overwrote the Amount fit).

Client sends raw, unscaled values for V1-V28, Amount, and Time - this
API does the scaling internally, exactly matching the training pipeline.
"""

import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud-inference-api")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

app = FastAPI(title="Fraud Detection Inference API", version="1.0.0")

model = None
scaler_amount = None
scaler_time = None
load_error: str | None = None


def load_artifacts() -> None:
    global model, scaler_amount, scaler_time, load_error
    try:
        model = joblib.load(MODEL_DIR / "fraud_xgb_model.pkl")
        scaler_amount = joblib.load(MODEL_DIR / "scaler_amount.pkl")
        scaler_time = joblib.load(MODEL_DIR / "scaler_time.pkl")
        logger.info("Model and both scalers loaded successfully")
    except Exception as exc:  # noqa: BLE001
        load_error = str(exc)
        logger.exception("Failed to load model/scalers")


load_artifacts()


class Transaction(BaseModel):
    v: list[float] = Field(..., min_length=28, max_length=28, description="V1..V28, in order")
    amount: float = Field(..., ge=0, description="Raw transaction amount, unscaled")
    time: float = Field(..., ge=0, description="Raw seconds since first transaction, unscaled")


class PredictResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    latency_ms: float


@app.get("/")
def root():
    return {"service": "fraud-inference-api", "status": "ok" if model is not None else "not_ready"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or "Model not loaded")
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    if model is None or scaler_amount is None or scaler_time is None:
        raise HTTPException(status_code=503, detail="Model or scalers not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(txn: Transaction, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or "Model not loaded")

    start = time.perf_counter()
    try:
        scaled_amount = scaler_amount.transform([[txn.amount]])[0][0]
        scaled_time = scaler_time.transform([[txn.time]])[0][0]

        features = np.array(txn.v + [scaled_amount, scaled_time]).reshape(1, -1)

        pred = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0][1])

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict client=%s latency_ms=%.2f prediction=%s proba=%.4f",
            request.client.host if request.client else "unknown",
            latency_ms,
            pred,
            proba,
        )
        return PredictResponse(
            prediction=pred,
            label="FRAUD" if pred == 1 else "LEGITIMATE",
            fraud_probability=round(proba, 4),
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


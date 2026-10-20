"""
Credit Card Fraud Detection project adapter.

Plugs into the shared inference server as a FastAPI router. Project-specific
parts (everything that used to live in the single-project main.py):
  1. the input columns by name: Time, V1..V28, Amount  - FIELDS below
  2. scaling: Amount and Time are scaled with their OWN scalers (the original
     single-scaler bug is fixed), then the 30 features are assembled in the
     order the model was trained on: V1..V28, scaled_amount, scaled_time
  3. the output labels FRAUD / LEGITIMATE

Files expected in <MODEL_DIR>/fraud/ :
    fraud_xgb_model.pkl, scaler_amount.pkl, scaler_time.pkl
"""

import logging
import os
import threading
import time
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.projects.common import build_schema

logger = logging.getLogger("fraud")

PROJECT_ID = "fraud"
TITLE = "Credit Card Fraud Detection"
DESCRIPTION = "Flags a card transaction as FRAUD or LEGITIMATE."

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")) / "fraud"
SCALER_AMOUNT_FILE = "scaler_amount.pkl"
SCALER_TIME_FILE = "scaler_time.pkl"

# ---- Models offered for this project (shown in the model dropdown) ---------
MODELS = {
    "xgboost": {"label": "XGBoost", "file": "fraud_xgb_model.pkl"},
}
ModelName = Enum("ModelName", {key: key for key in MODELS}, type=str)

# ---- Input columns, as they appear in the dataset ---------------------------
V_NAMES = [f"V{i}" for i in range(1, 29)]

FIELDS = (
    [dict(name="Time", kind="float", required=True, min=0, example=40000.0,
          desc="Raw seconds elapsed since the first transaction (unscaled)")]
    + [dict(name=n, kind="float", required=True,
            desc=f"Anonymized PCA component {n[1:]}") for n in V_NAMES]
    + [dict(name="Amount", kind="float", required=True, min=0, example=50.0,
            desc="Raw transaction amount (unscaled)")]
)

Transaction = build_schema("Transaction", FIELDS)

# ---- Example inputs for the UI's example buttons ----------------------------
# These are synthetic values (the dataset's V columns are anonymized), chosen so
# you can show both outcomes in a demo.
SAMPLES = [
    dict(name="Baseline (all V = 0)",
         values={"Time": 40000, "Amount": 50, **{n: 0.0 for n in V_NAMES}}),
    dict(name="Model flags as fraud",
         values={"Time": 94813, "Amount": 120.0,
                 "V1": 0.55, "V2": 0.51, "V3": 0.28, "V4": 3.29, "V5": 0.18, "V6": -0.6,
                 "V7": -1.01, "V8": -0.4, "V9": 0.59, "V10": -6.52, "V11": 0.98,
                 "V12": -9.37, "V13": 0.78, "V14": -5.82, "V15": 0.18, "V16": 0.77,
                 "V17": 1.77, "V18": -0.33, "V19": -0.69, "V20": 1.96, "V21": 0.24,
                 "V22": 0.97, "V23": 0.87, "V24": -1.14, "V25": 2.43, "V26": 1.78,
                 "V27": 0.18, "V28": 0.75}),
]


class PredictResponse(BaseModel):
    project: str
    chosen_model: str
    prediction: int
    label: str
    fraud_probability: float
    latency_ms: float


def missing_files() -> list[str]:
    needed = [m["file"] for m in MODELS.values()] + [SCALER_AMOUNT_FILE, SCALER_TIME_FILE]
    return [name for name in needed if not (MODEL_DIR / name).exists()]


# ---- Lazy loading (the server has little RAM, so load only when first used) -
_lock = threading.Lock()
_cache: dict = {}


def _load(model_key: str):
    with _lock:
        if model_key in _cache:
            return _cache[model_key]
        try:
            model = joblib.load(MODEL_DIR / MODELS[model_key]["file"])
            scaler_amount = joblib.load(MODEL_DIR / SCALER_AMOUNT_FILE)
            scaler_time = joblib.load(MODEL_DIR / SCALER_TIME_FILE)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not load fraud model '%s'", model_key)
            raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
        _cache[model_key] = (model, scaler_amount, scaler_time)
        logger.info("Loaded fraud model '%s' and both scalers", model_key)
        return _cache[model_key]


def _run(model_key: str, txn, request: Request) -> PredictResponse:
    model, scaler_amount, scaler_time = _load(model_key)

    start = time.perf_counter()
    try:
        scaled_amount = scaler_amount.transform([[txn.Amount]])[0][0]
        scaled_time = scaler_time.transform([[txn.Time]])[0][0]

        # Rebuild V1..V28 in order, then append the two scaled values.
        # Feature order matches training: V1..V28, scaled_amount, scaled_time.
        v_values = [getattr(txn, name) for name in V_NAMES]
        features = np.array(v_values + [scaled_amount, scaled_time]).reshape(1, -1)

        pred = int(model.predict(features)[0])
        proba = float(model.predict_proba(features)[0][1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fraud prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "fraud model=%s client=%s latency_ms=%.2f prediction=%s proba=%.4f",
        model_key,
        request.client.host if request.client else "unknown",
        latency_ms, pred, proba,
    )
    return PredictResponse(
        project=PROJECT_ID,
        chosen_model=model_key,
        prediction=pred,
        label="FRAUD" if pred == 1 else "LEGITIMATE",
        fraud_probability=round(proba, 4),
        latency_ms=round(latency_ms, 2),
    )


# ---- Routes -----------------------------------------------------------------
router = APIRouter(prefix=f"/{PROJECT_ID}", tags=["Credit Card Fraud"])


@router.get("/models")
def list_models():
    return [{"id": key, "label": meta["label"]} for key, meta in MODELS.items()]


@router.get("/fields")
def list_fields():
    """Column definitions, so a UI can draw the form without hard-coding it."""
    return FIELDS


@router.post("/predict/{model_name}", response_model=PredictResponse)
def predict(model_name: ModelName, txn: Transaction, request: Request):  # type: ignore[valid-type]
    return _run(model_name.value, txn, request)


# Old single-project route (POST /predict), kept so load_test.py and anything
# else that still calls it keeps working. Hidden from /docs; remove when unused.
legacy_router = APIRouter()


@legacy_router.post("/predict", response_model=PredictResponse, include_in_schema=False)
def predict_legacy(txn: Transaction, request: Request):  # type: ignore[valid-type]
    return _run("xgboost", txn, request)


"""
Employee Attrition project adapter.

Plugs into the shared inference server as a FastAPI router. It knows three
things that are specific to this project:
  1. the input columns (named, with allowed values) - FIELDS below
  2. how to turn them into the 43 model features, exactly as the training
     notebook did (label maps + one-hot with drop_first + StandardScaler)
  3. the output labels and the tuned decision threshold (0.3, stored in the pkl)

Files expected in <MODEL_DIR>/attrition/ :
    best_model_xgb.pkl   dict: {'model', 'threshold', 'feature_names'}
    scaler.pkl           StandardScaler fitted on the 43 training features
"""

import logging
import os
import pickle
import threading
import time
from enum import Enum
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.projects.common import build_schema

logger = logging.getLogger("attrition")

PROJECT_ID = "attrition"
TITLE = "Employee Attrition"
DESCRIPTION = "Estimates whether an employee is at high or low risk of leaving."

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")) / "attrition"
SCALER_FILE = "scaler.pkl"

# ---- Models offered for this project (shown in the model dropdown) ---------
# Only XGBoost was saved by the training notebook, so it is the only entry.
MODELS = {
    "xgboost": {
        "label": "XGBoost (tuned decision threshold 0.3)",
        "file": "best_model_xgb.pkl",
    },
}
ModelName = Enum("ModelName", {key: key for key in MODELS}, type=str)

# ---- Input columns: one row per column, in dataset order -------------------
# kind: "int" | "float" | "choice".  Defaults are the same values the Streamlit
# app uses for the fields its form does not collect.
FIELDS = [
    dict(name="Age", kind="int", default=30, min=18, max=60, desc="Age in years"),
    dict(name="BusinessTravel", kind="choice", default="Non-Travel",
         choices=["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
         desc="How often the employee travels for work"),
    dict(name="DailyRate", kind="float", default=800, min=0, desc="Daily rate"),
    dict(name="Department", kind="choice", default="Sales",
         choices=["Sales", "Research & Development", "Human Resources"],
         desc="Department"),
    dict(name="DistanceFromHome", kind="int", default=5, min=1, max=30,
         desc="Distance from home to office (km)"),
    dict(name="Education", kind="int", default=3, min=1, max=5,
         desc="Education level, 1 (lowest) to 5 (highest)"),
    dict(name="EducationField", kind="choice", default="Life Sciences",
         choices=["Life Sciences", "Medical", "Marketing", "Technical Degree",
                  "Other", "Human Resources"],
         desc="Field of education"),
    dict(name="EnvironmentSatisfaction", kind="int", default=3, min=1, max=4,
         desc="1 (low) to 4 (very high)"),
    dict(name="Gender", kind="choice", default="Male", choices=["Male", "Female"],
         desc="Gender"),
    dict(name="HourlyRate", kind="float", default=65, min=0, desc="Hourly rate"),
    dict(name="JobInvolvement", kind="int", default=3, min=1, max=4,
         desc="1 (low) to 4 (very high)"),
    dict(name="JobLevel", kind="int", default=2, min=1, max=5,
         desc="Job level, 1 to 5"),
    dict(name="JobRole", kind="choice", default="Sales Executive",
         choices=["Sales Executive", "Research Scientist", "Laboratory Technician",
                  "Manufacturing Director", "Healthcare Representative", "Manager",
                  "Sales Representative", "Research Director", "Human Resources"],
         desc="Job role"),
    dict(name="JobSatisfaction", kind="int", default=3, min=1, max=4,
         desc="1 (low) to 4 (very high)"),
    dict(name="MaritalStatus", kind="choice", default="Single",
         choices=["Single", "Married", "Divorced"], desc="Marital status"),
    dict(name="MonthlyIncome", kind="float", default=5000, min=1000, max=20000,
         desc="Monthly income"),
    dict(name="MonthlyRate", kind="float", default=14000, min=0, desc="Monthly rate"),
    dict(name="NumCompaniesWorked", kind="int", default=2, min=0,
         desc="Number of companies worked at before"),
    dict(name="OverTime", kind="choice", default="No", choices=["No", "Yes"],
         desc="Works overtime?"),
    dict(name="PercentSalaryHike", kind="int", default=15, min=0,
         desc="Last salary hike (%)"),
    dict(name="PerformanceRating", kind="int", default=3, min=1, max=4,
         desc="1 (low) to 4 (outstanding)"),
    dict(name="RelationshipSatisfaction", kind="int", default=3, min=1, max=4,
         desc="1 (low) to 4 (very high)"),
    dict(name="StockOptionLevel", kind="int", default=1, min=0, max=3,
         desc="Stock option level, 0 to 3"),
    dict(name="TotalWorkingYears", kind="int", default=5, min=0, max=40,
         desc="Total years of work experience"),
    dict(name="TrainingTimesLastYear", kind="int", default=2, min=0,
         desc="Training sessions attended last year"),
    dict(name="WorkLifeBalance", kind="int", default=3, min=1, max=4,
         desc="1 (bad) to 4 (best)"),
    dict(name="YearsAtCompany", kind="int", default=3, min=0, max=40,
         desc="Years at the company"),
    dict(name="YearsInCurrentRole", kind="int", default=3, min=0, max=40,
         desc="Years in the current role"),
    dict(name="YearsSinceLastPromotion", kind="int", default=1, min=0, max=40,
         desc="Years since last promotion"),
    dict(name="YearsWithCurrManager", kind="int", default=3, min=0, max=40,
         desc="Years with the current manager"),
]


EmployeeInput = build_schema("EmployeeInput", FIELDS)


class PredictResponse(BaseModel):
    project: str
    chosen_model: str
    prediction: int
    label: str
    attrition_probability: float
    threshold: float
    latency_ms: float
    fields_defaulted: list[str]


def missing_files() -> list[str]:
    needed = [m["file"] for m in MODELS.values()] + [SCALER_FILE]
    return [name for name in needed if not (MODEL_DIR / name).exists()]


# ---- Encoding: mirrors 02_preprocessing.ipynb ------------------------------
ORDINAL_MAPS = {
    "Gender": {"Male": 1, "Female": 0},
    "OverTime": {"Yes": 1, "No": 0},
    "BusinessTravel": {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2},
}
ONEHOT_FIELDS = {"Department", "EducationField", "JobRole", "MaritalStatus"}


def encode(values: dict, feature_names: list[str]) -> pd.DataFrame:
    """Build the one-row feature table in the exact column order of training."""
    row = {}
    for col in feature_names:
        if col in ORDINAL_MAPS:
            row[col] = ORDINAL_MAPS[col][values[col]]
        elif col in values:
            row[col] = values[col]
        else:
            # A one-hot dummy such as 'JobRole_Manager' -> field 'JobRole', category 'Manager'.
            # The category dropped by drop_first has no column, so it ends up as all zeros.
            field, _, category = col.partition("_")
            if field not in ONEHOT_FIELDS:
                raise ValueError(f"Unknown model feature: {col}")
            row[col] = 1 if values[field] == category else 0
    return pd.DataFrame([row], columns=feature_names).astype(float)


# ---- Lazy loading (the server has little RAM, so load only when first used) -
_lock = threading.Lock()
_cache: dict = {}


def _load(model_key: str):
    with _lock:
        if model_key in _cache:
            return _cache[model_key]
        try:
            with open(MODEL_DIR / MODELS[model_key]["file"], "rb") as fh:
                package = pickle.load(fh)
            with open(MODEL_DIR / SCALER_FILE, "rb") as fh:
                scaler = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not load attrition model '%s'", model_key)
            raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
        _cache[model_key] = (package, scaler)
        logger.info("Loaded attrition model '%s'", model_key)
        return _cache[model_key]


# ---- Routes -----------------------------------------------------------------
router = APIRouter(prefix=f"/{PROJECT_ID}", tags=["Employee Attrition"])


@router.get("/models")
def list_models():
    return [{"id": key, "label": meta["label"]} for key, meta in MODELS.items()]


@router.get("/fields")
def list_fields():
    """Column definitions, so a UI can draw the form without hard-coding it."""
    return FIELDS


@router.post("/predict/{model_name}", response_model=PredictResponse)
def predict(model_name: ModelName, employee: EmployeeInput):  # type: ignore[valid-type]
    key = model_name.value
    package, scaler = _load(key)
    model = package["model"]
    threshold = package["threshold"]
    feature_names = package["feature_names"]

    start = time.perf_counter()
    try:
        values = employee.model_dump()
        features = encode(values, feature_names)
        scaled = pd.DataFrame(scaler.transform(features), columns=feature_names)
        probability = float(model.predict_proba(scaled)[0][1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Attrition prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    prediction = 1 if probability >= threshold else 0
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("attrition model=%s proba=%.4f pred=%s latency_ms=%.2f",
                key, probability, prediction, latency_ms)
    supplied = employee.model_fields_set
    return PredictResponse(
        project="attrition",
        chosen_model=key,
        prediction=prediction,
        label="HIGH_RISK" if prediction == 1 else "LOW_RISK",
        attrition_probability=round(probability, 4),
        threshold=threshold,
        latency_ms=round(latency_ms, 2),
        fields_defaulted=[f["name"] for f in FIELDS if f["name"] not in supplied],
    )
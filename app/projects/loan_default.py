"""
Loan Default Risk project adapter (Lending Club).

Plugs into the shared inference server as a FastAPI router. The project-specific
parts are:
  1. the input columns by name (39 of them) - FIELDS below
  2. turning them into the 47 features the models were trained on, exactly as
     02_feature_engineering.ipynb did: 8 columns are DERIVED here (ratios, flags,
     fico_range_high), so the caller never has to compute them
  3. the 8 text columns are passed as pandas 'category' columns, which is how
     LightGBM was trained (the model remembers its own category lists)
  4. five saved models, each with its own decision threshold

Files expected in <MODEL_DIR>/loan_default/ (all from the notebooks' models/ folder):
    calibrated_platt.pkl, calibrated_isotonic.pkl, tuned_lightgbm.pkl,
    scale_pos_weight_lightgbm.pkl, baseline_lightgbm.pkl
"""

import logging
import os
import threading
import time
from enum import Enum
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.projects.common import build_schema

logger = logging.getLogger("loan_default")

PROJECT_ID = "loan_default"
TITLE = "Loan Default Risk"
DESCRIPTION = "Estimates the chance that a Lending Club loan applicant defaults."

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")) / PROJECT_ID

# ---- Models offered for this project (shown in the model dropdown) ---------
# threshold = probability at or above which the loan is flagged HIGH_RISK.
#  - calibrated models give real default probabilities (base rate is about 20%),
#    so 0.2 is used, the value your KS plot (08_evaluation_credit_risk) points to.
#  - the other three keep the 0.5 your notebooks evaluated them at.
MODELS = {
    "tuned_platt": {
        "label": "Tuned LightGBM + Platt calibration (final model)",
        "file": "calibrated_platt.pkl", "threshold": 0.2,
    },
    "tuned_isotonic": {
        "label": "Tuned LightGBM + isotonic calibration",
        "file": "calibrated_isotonic.pkl", "threshold": 0.2,
    },
    "tuned": {
        "label": "Tuned LightGBM (uncalibrated)",
        "file": "tuned_lightgbm.pkl", "threshold": 0.5,
    },
    "scale_pos_weight": {
        "label": "LightGBM with scale_pos_weight",
        "file": "scale_pos_weight_lightgbm.pkl", "threshold": 0.5,
    },
    "baseline": {
        "label": "Baseline LightGBM (default settings)",
        "file": "baseline_lightgbm.pkl", "threshold": 0.5,
    },
}
ModelName = Enum("ModelName", {key: key for key in MODELS}, type=str)

# ---- Input columns (defaults = typical values from the training data) ------
STATES = ("AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO "
          "MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY").split()

FIELDS = [
    dict(name="loan_amnt", kind="float", default=12000, min=500, max=40000, desc="Loan amount requested ($)"),
    dict(name="term", kind="choice", default="36 months", choices=["36 months", "60 months"], desc="Loan term"),
    dict(name="installment", kind="float", default=375, min=0, desc="Monthly installment ($)"),
    dict(name="emp_length", kind="choice", default="10+ years",
         choices=["< 1 year", "1 year"] + [f"{n} years" for n in range(2, 10)] + ["10+ years", "Unknown"],
         desc="Length of employment"),
    dict(name="home_ownership", kind="choice", default="MORTGAGE",
         choices=["MORTGAGE", "RENT", "OWN", "ANY", "OTHER"], desc="Home ownership"),
    dict(name="annual_inc", kind="float", default=65000, min=0, desc="Annual income ($)"),
    dict(name="verification_status", kind="choice", default="Source Verified",
         choices=["Not Verified", "Source Verified", "Verified"], desc="Income verification"),
    dict(name="purpose", kind="choice", default="debt_consolidation",
         choices=["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                  "medical", "small_business", "car", "moving", "vacation", "house", "wedding",
                  "renewable_energy", "educational"], desc="Purpose of the loan"),
    dict(name="addr_state", kind="choice", default="CA", choices=STATES, desc="US state of the applicant"),
    dict(name="dti", kind="float", default=17.6, min=0, max=999, desc="Debt-to-income ratio (%)"),
    dict(name="fico_range_low", kind="float", default=690, min=625, max=845,
         desc="FICO score (lower end of the applicant's band)"),
    dict(name="inq_last_6mths", kind="int", default=0, min=0, max=8, desc="Credit inquiries in the last 6 months"),
    dict(name="mths_since_last_delinq", kind="int", default=999, min=0, max=999,
         desc="Months since last delinquency (999 = never)"),
    dict(name="mths_since_last_record", kind="int", default=999, min=0, max=999,
         desc="Months since last public record (999 = never)"),
    dict(name="open_acc", kind="int", default=11, min=0, desc="Open credit lines"),
    dict(name="pub_rec", kind="int", default=0, min=0, desc="Derogatory public records"),
    dict(name="revol_bal", kind="float", default=11134, min=0, desc="Total revolving balance ($)"),
    dict(name="revol_util", kind="float", default=52.2, min=0, desc="Revolving line utilization (%)"),
    dict(name="total_acc", kind="int", default=23, min=0, desc="Total credit lines ever opened"),
    dict(name="initial_list_status", kind="choice", default="w", choices=["w", "f"],
         desc="Initial listing status of the loan"),
    dict(name="application_type", kind="choice", default="Individual", choices=["Individual", "Joint App"],
         desc="Individual or joint application"),
    dict(name="acc_now_delinq", kind="int", default=0, min=0, desc="Accounts currently delinquent"),
    dict(name="tot_coll_amt", kind="float", default=0, min=0, desc="Total collection amounts ever owed ($)"),
    dict(name="tot_cur_bal", kind="float", default=80231, min=0, desc="Total current balance, all accounts ($)"),
    dict(name="avg_cur_bal", kind="float", default=7407, min=0, desc="Average current balance, all accounts ($)"),
    dict(name="bc_open_to_buy", kind="float", default=4700, min=0, desc="Total open to buy on bankcards ($)"),
    dict(name="bc_util", kind="float", default=63.2, min=0, desc="Bankcard utilization (%)"),
    dict(name="mo_sin_old_rev_tl_op", kind="int", default=164, min=0,
         desc="Months since oldest revolving account was opened"),
    dict(name="mort_acc", kind="int", default=1, min=0, desc="Number of mortgage accounts"),
    dict(name="num_actv_bc_tl", kind="int", default=3, min=0, desc="Active bankcard accounts"),
    dict(name="num_actv_rev_tl", kind="int", default=5, min=0, desc="Active revolving accounts"),
    dict(name="num_tl_op_past_12m", kind="int", default=2, min=0, desc="Accounts opened in the past 12 months"),
    dict(name="pct_tl_nvr_dlq", kind="float", default=98, min=0, max=100, desc="Percent of accounts never delinquent"),
    dict(name="percent_bc_gt_75", kind="float", default=42.9, min=0, max=100,
         desc="Percent of bankcards above 75% of their limit"),
    dict(name="pub_rec_bankruptcies", kind="int", default=0, min=0, desc="Public record bankruptcies"),
    dict(name="tax_liens", kind="int", default=0, min=0, desc="Number of tax liens"),
    dict(name="total_bc_limit", kind="float", default=15100, min=0, desc="Total bankcard credit limit ($)"),
    dict(name="total_il_high_credit_limit", kind="float", default=31681, min=0,
         desc="Total installment high credit limit ($)"),
    dict(name="credit_history_years", kind="float", default=15.8, min=0, max=90,
         desc="Years between the first credit line and this loan"),
]

LoanApplication = build_schema("LoanApplication", FIELDS)

# ---- Example inputs for the UI's example buttons ----------------------------
# Only the columns that differ from the defaults above are listed. These describe
# a stronger and a weaker credit profile; they are not taken from real borrowers.
SAMPLES = [
    dict(name="Stronger credit profile",
         values={"loan_amnt": 8000, "installment": 260, "annual_inc": 120000, "dti": 8,
                 "fico_range_low": 770, "revol_util": 15, "bc_util": 12, "revol_bal": 6000,
                 "total_acc": 30, "credit_history_years": 25, "mort_acc": 3,
                 "pct_tl_nvr_dlq": 100, "percent_bc_gt_75": 0, "tot_cur_bal": 250000,
                 "avg_cur_bal": 25000, "bc_open_to_buy": 25000, "total_bc_limit": 40000,
                 "total_il_high_credit_limit": 60000, "num_tl_op_past_12m": 1}),
    dict(name="Weaker credit profile",
         values={"term": "60 months", "loan_amnt": 28000, "installment": 700, "annual_inc": 32000,
                 "dti": 34, "fico_range_low": 640, "inq_last_6mths": 4, "mths_since_last_delinq": 8,
                 "mths_since_last_record": 40, "revol_util": 92, "bc_util": 96, "revol_bal": 21000,
                 "total_acc": 12, "credit_history_years": 5, "mort_acc": 0, "pub_rec": 1,
                 "pub_rec_bankruptcies": 1, "pct_tl_nvr_dlq": 80, "percent_bc_gt_75": 90,
                 "tot_cur_bal": 20000, "avg_cur_bal": 1500, "bc_open_to_buy": 300,
                 "total_bc_limit": 8000, "total_il_high_credit_limit": 12000,
                 "num_tl_op_past_12m": 6, "emp_length": "< 1 year", "home_ownership": "RENT",
                 "purpose": "small_business", "verification_status": "Verified"}),
]

# ---- Feature building: mirrors 01_eda.ipynb + 02_feature_engineering.ipynb ---
# The 47 model features, in training order.
FEATURE_ORDER = [
    "loan_amnt", "term", "installment", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "dti", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status", "application_type",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "mo_sin_old_rev_tl_op", "mort_acc", "num_actv_bc_tl", "num_actv_rev_tl", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "total_bc_limit",
    "total_il_high_credit_limit", "never_delinq", "never_public_record", "installment_to_income",
    "loan_to_income", "credit_history_years", "revol_bal_to_income", "high_utilization_flag",
    "recent_credit_seeking_flag",
]
# Text columns, in the order the model stored their category lists.
CATEGORICAL = ["term", "emp_length", "home_ownership", "verification_status", "purpose",
               "addr_state", "initial_list_status", "application_type"]

# Caps the notebooks applied to the training data.
INCOME_CAP = 250000.0                      # annual_inc clipped at its 99th percentile
DTI_CAP = 65.26651                         # dti clipped at its 99.9th percentile
RATIO_CAPS = {                             # 99th-percentile caps of the three ratio features
    "installment_to_income": 0.1993550828936155,
    "loan_to_income": 0.5,
    "revol_bal_to_income": 0.9571979,
}


def _capped_ratio(numerator: float, income: float, cap: float, months: int = 1) -> float:
    """numerator / (income / months), with the notebooks' handling: zero income -> cap, then clip at cap."""
    if income <= 0:
        return cap
    return min(numerator / (income / months), cap)


def build_features(v: dict) -> tuple[dict, dict]:
    """Raw columns -> the 47 model features. Also returns the derived values for display."""
    income = min(v["annual_inc"], INCOME_CAP)
    low = v["fico_range_low"]
    derived = {
        "fico_range_high": 850 if low >= 845 else low + 4,
        "never_delinq": int(v["mths_since_last_delinq"] >= 999),
        "never_public_record": int(v["mths_since_last_record"] >= 999),
        "installment_to_income": _capped_ratio(v["installment"], income, RATIO_CAPS["installment_to_income"], months=12),
        "loan_to_income": _capped_ratio(v["loan_amnt"], income, RATIO_CAPS["loan_to_income"]),
        "revol_bal_to_income": _capped_ratio(v["revol_bal"], income, RATIO_CAPS["revol_bal_to_income"]),
        "high_utilization_flag": int(v["bc_util"] > 90),
        "recent_credit_seeking_flag": int(v["num_tl_op_past_12m"] >= 5 or v["inq_last_6mths"] >= 3),
    }
    row = {**v, **derived}
    row["annual_inc"] = income
    row["dti"] = min(v["dti"], DTI_CAP)
    return row, derived


# ---- Loading: finds the LightGBM model inside the saved object --------------
class _Bundle:
    def __init__(self, estimator):
        self.estimator = estimator
        base = estimator
        if hasattr(estimator, "calibrated_classifiers_"):          # CalibratedClassifierCV(FrozenEstimator(lgbm))
            base = estimator.calibrated_classifiers_[0].estimator
            base = getattr(base, "estimator", base)                 # unwrap FrozenEstimator
        names = list(base.feature_name_)
        if sorted(names) != sorted(FEATURE_ORDER):
            raise ValueError(f"model features differ from this adapter: {sorted(set(names) ^ set(FEATURE_ORDER))}")
        self.feature_order = names                                  # trust the model's own order
        self.categories = dict(zip(CATEGORICAL, base.booster_.pandas_categorical))
        for field in FIELDS:                                        # warn about choices the model never saw
            if field["name"] in self.categories:
                seen = {str(c).strip() for c in self.categories[field["name"]]}
                unseen = [c for c in field["choices"] if c.strip() not in seen]
                if unseen:
                    logger.warning("%s: choices not seen in training: %s", field["name"], unseen)


_lock = threading.Lock()
_cache: dict = {}


def _load(model_key: str) -> _Bundle:
    with _lock:
        if model_key in _cache:
            return _cache[model_key]
        try:
            bundle = _Bundle(joblib.load(MODEL_DIR / MODELS[model_key]["file"]))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not load loan model '%s'", model_key)
            raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
        _cache[model_key] = bundle
        logger.info("Loaded loan model '%s'", model_key)
        return bundle


def _training_category(column: str, value: str, categories: list) -> str:
    """The model stored ' 36 months' (leading space); the API accepts '36 months'. Match ignoring spaces."""
    wanted = value.strip()
    for c in categories:
        if str(c).strip() == wanted:
            return c
    raise ValueError(f"'{value}' is not a {column} value the model saw in training")


def to_frame(row: dict, bundle: _Bundle) -> pd.DataFrame:
    data = {}
    for col in bundle.feature_order:
        if col in CATEGORICAL:
            cats = bundle.categories[col]
            data[col] = pd.Categorical([_training_category(col, row[col], cats)], categories=cats)
        else:
            data[col] = [float(row[col])]
    return pd.DataFrame(data)


def missing_files() -> list[str]:
    return [m["file"] for m in MODELS.values() if not (MODEL_DIR / m["file"]).exists()]


class PredictResponse(BaseModel):
    project: str
    chosen_model: str
    prediction: int
    label: str
    default_probability: float
    threshold: float
    latency_ms: float
    derived: dict[str, float]
    fields_defaulted: list[str]


# ---- Routes -----------------------------------------------------------------
router = APIRouter(prefix=f"/{PROJECT_ID}", tags=["Loan Default"])


@router.get("/models")
def list_models():
    return [{"id": key, "label": meta["label"]} for key, meta in MODELS.items()]


@router.get("/fields")
def list_fields():
    """Column definitions, so a UI can draw the form without hard-coding it."""
    return FIELDS


@router.post("/predict/{model_name}", response_model=PredictResponse)
def predict(model_name: ModelName, applicant: LoanApplication):  # type: ignore[valid-type]
    key = model_name.value
    bundle = _load(key)

    start = time.perf_counter()
    try:
        row, derived = build_features(applicant.model_dump())
        probability = float(bundle.estimator.predict_proba(to_frame(row, bundle))[0][1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Loan prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    threshold = MODELS[key]["threshold"]
    prediction = 1 if probability >= threshold else 0
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("loan model=%s proba=%.4f pred=%s latency_ms=%.2f", key, probability, prediction, latency_ms)
    supplied = applicant.model_fields_set
    return PredictResponse(
        project=PROJECT_ID,
        chosen_model=key,
        prediction=prediction,
        label="HIGH_RISK" if prediction == 1 else "LOW_RISK",
        default_probability=round(probability, 4),
        threshold=threshold,
        latency_ms=round(latency_ms, 2),
        derived={k: round(float(x), 4) for k, x in derived.items()},
        fields_defaulted=[f["name"] for f in FIELDS if f["name"] not in supplied],
    )


"""
Financial News Sentiment project adapter.

Plugs into the shared inference server as a FastAPI router. It knows what is
specific to this project:
  1. the one input column, free text (a news headline or sentence)
  2. how to clean the text exactly as the training notebook did
     (lowercase, strip punctuation, NLTK tokenise, drop stopwords, lemmatise)
  3. how to turn the text into TF-IDF features and read the three class
     probabilities: negative / neutral / positive

Files expected in <MODEL_DIR>/sentiment/ :
    sentiment_model.pkl     CalibratedClassifierCV(LinearSVC), classes negative/neutral/positive
    tfidf_vectorizer.pkl    TfidfVectorizer(max_features=5000), shared by every model below

NLTK data needed on the machine (one-time download, no network at run time):
    python -m nltk.downloader stopwords wordnet punkt_tab
"""

import logging
import os
import re
import threading
import time
from enum import Enum
from pathlib import Path

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.projects.common import build_schema

logger = logging.getLogger("sentiment")

PROJECT_ID = "sentiment"
TITLE = "Financial News Sentiment"
DESCRIPTION = "Reads a financial news headline and says whether the tone is positive, neutral or negative."

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")) / "sentiment"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# ---- Models offered for this project (shown in the model dropdown) ---------
# Every model here must have predict_proba and be trained on the same TF-IDF
# vectorizer. The notebook also trained Naive Bayes and Logistic Regression but
# only the calibrated SVM was saved; once saved, add them as extra lines here.
MODELS = {
    "calibrated_svm": {
        "label": "Calibrated linear SVM (TF-IDF)",
        "file": "sentiment_model.pkl",
    },
}
ModelName = Enum("ModelName", {key: key for key in MODELS}, type=str)

# NLTK data the cleaning step needs: name -> paths nltk.data.find() may find it at.
# WordNet is often left as a .zip, which NLTK reads directly.
NLTK_RESOURCES = {
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "punkt_tab": ("tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"),
}
NLTK_HELP = "python -m nltk.downloader stopwords wordnet punkt_tab"

# ---- Input column ------------------------------------------------------------
FIELDS = [
    dict(name="text", kind="text", required=True, max_length=2000,
         example="Operating profit rose to EUR 12.5 million from EUR 9.1 million.",
         desc="A financial news headline or sentence, in English"),
]

NewsInput = build_schema("NewsInput", FIELDS)

# ---- Example inputs for the UI's example buttons ----------------------------
SAMPLES = [
    dict(name="Positive headline",
         values={"text": "Operating profit rose to EUR 12.5 million from EUR 9.1 million as sales climbed 18%, "
                         "beating analyst expectations."}),
    dict(name="Negative headline",
         values={"text": "The company issued a profit warning, will cut 300 jobs and saw its shares plunge "
                         "after quarterly losses widened."}),
    dict(name="Neutral headline",
         values={"text": "The annual general meeting will be held on 15 April at the company's headquarters in Helsinki."}),
]

# How the UI colours each outcome. Unknown labels fall back to neutral.
TONE = {"positive": "ok", "negative": "bad", "neutral": "neutral"}


class PredictResponse(BaseModel):
    project: str
    chosen_model: str
    prediction: str
    label: str
    tone: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    words_in_text: int
    words_recognized: int
    latency_ms: float


def _nltk_missing() -> list[str]:
    """Names of NLTK resources that are not installed (also covers nltk itself)."""
    try:
        from nltk.data import find
    except ImportError:
        return ["nltk (python package)"]
    missing = []
    for name, paths in NLTK_RESOURCES.items():
        for path in paths:
            try:
                find(path)
                break
            except LookupError:
                continue
        else:
            missing.append(f"nltk:{name}")
    return missing


def missing_files() -> list[str]:
    needed = [m["file"] for m in MODELS.values()] + [VECTORIZER_FILE]
    return [n for n in needed if not (MODEL_DIR / n).exists()] + _nltk_missing()


# ---- Text cleaning: the same steps as src/preprocessing.py in the project repo
def _build_cleaner():
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # NLTK loads its corpora lazily on first use and that first load is not
    # thread-safe, so touch everything once here (we hold the lock) before
    # real requests arrive on several threads.
    lemmatizer.lemmatize("warm")
    word_tokenize("warm up")

    def clean_text(text: str) -> str:
        text = text.lower()
        text = text.replace("-", " ")
        text = re.sub(r"[^a-z0-9.\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [w.strip(".") for w in tokens]
        tokens = [w for w in tokens if w]
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    return clean_text


# ---- Lazy loading (the server has little RAM, so load only when first used) -
_lock = threading.Lock()
_cache: dict = {}   # model key -> fitted classifier
_shared: dict = {}  # "vectorizer", "clean" - the same for every model


def _load(model_key: str):
    with _lock:
        try:
            if "vectorizer" not in _shared:
                gone = _nltk_missing()
                if gone:
                    raise RuntimeError(f"missing {', '.join(gone)}. Run: {NLTK_HELP}")
                _shared["vectorizer"] = joblib.load(MODEL_DIR / VECTORIZER_FILE)
                _shared["clean"] = _build_cleaner()
            if model_key not in _cache:
                model = joblib.load(MODEL_DIR / MODELS[model_key]["file"])
                if not hasattr(model, "predict_proba"):
                    raise RuntimeError("this model has no predict_proba, so it cannot give class probabilities")
                _cache[model_key] = model
                logger.info("Loaded sentiment model '%s'", model_key)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not load sentiment model '%s'", model_key)
            raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
        return _cache[model_key], _shared["vectorizer"], _shared["clean"]


# ---- Routes -----------------------------------------------------------------
router = APIRouter(prefix=f"/{PROJECT_ID}", tags=["Financial News Sentiment"])


@router.get("/models")
def list_models():
    return [{"id": key, "label": meta["label"]} for key, meta in MODELS.items()]


@router.get("/fields")
def list_fields():
    """Column definitions, so a UI can draw the form without hard-coding it."""
    return FIELDS


@router.post("/predict/{model_name}", response_model=PredictResponse)
def predict(model_name: ModelName, news: NewsInput):  # type: ignore[valid-type]
    key = model_name.value
    model, vectorizer, clean = _load(key)

    start = time.perf_counter()
    try:
        cleaned = clean(news.text)
        features = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(features)[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sentiment prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    # Two cases where the model would only be guessing: tell the caller instead.
    if not cleaned:
        raise HTTPException(status_code=422, detail="The text has no usable words after cleaning "
                                                    "(only stopwords, numbers or symbols). Please write a full sentence.")
    if features.nnz == 0:
        raise HTTPException(status_code=422, detail="None of the words in this text are known to the model "
                                                    "(it only knows its 5000 most useful words). Try a financial news sentence.")

    classes = [str(c) for c in model.classes_]
    best = max(range(len(classes)), key=lambda i: probabilities[i])
    top = classes[best]
    latency_ms = (time.perf_counter() - start) * 1000
    # The text itself is never logged, only its size.
    logger.info("sentiment model=%s chars=%d pred=%s conf=%.4f latency_ms=%.2f",
                key, len(news.text), top, probabilities[best], latency_ms)
    return PredictResponse(
        project=PROJECT_ID,
        chosen_model=key,
        prediction=top,
        label=top.upper(),
        tone=TONE.get(top, "neutral"),
        confidence=round(float(probabilities[best]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, probabilities)},
        cleaned_text=cleaned,
        words_in_text=len(set(cleaned.split())),
        words_recognized=int(features.nnz),
        latency_ms=round(latency_ms, 2),
    )


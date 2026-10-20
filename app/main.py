"""
ML Inference API - one server, many projects.

This file is the shared part and knows nothing about any specific model.
Each project lives in its own file under app/projects/ and exposes a router
plus a few constants (PROJECT_ID, TITLE, DESCRIPTION, MODELS, FIELDS, SAMPLES).

To add a project: write app/projects/<name>.py, import it below, and add it
to PROJECTS. Nothing else in this file changes.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.projects import attrition, fraud_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Every project the server offers.
PROJECTS = [fraud_detection, attrition]

app = FastAPI(
    title="ML Inference API",
    version="2.0.0",
    description="One server for several ML projects. Pick a project and a model, send named columns.",
)

for project in PROJECTS:
    app.include_router(project.router)
app.include_router(fraud_detection.legacy_router)  # old POST /predict


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def root():
    return {
        "service": "ml-inference-api",
        "projects": [p.PROJECT_ID for p in PROJECTS],
        "ui": "/ui",
        "docs": "/docs",
    }


@app.get("/ui", include_in_schema=False)
def ui():
    """The form page: pick a project and a model, fill in the columns."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Liveness: the process is up. Does not depend on any model."""
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    """Readiness: every project's model files are present on disk."""
    missing = {p.PROJECT_ID: m for p in PROJECTS if (m := p.missing_files())}
    if missing:
        raise HTTPException(status_code=503, detail={"missing_files": missing})
    return {"status": "ready", "projects": [p.PROJECT_ID for p in PROJECTS]}


@app.get("/projects")
def list_projects():
    """Everything a UI needs: projects, their models, and where to find the columns."""
    return [
        {
            "id": p.PROJECT_ID,
            "title": p.TITLE,
            "description": p.DESCRIPTION,
            "models": [{"id": k, "label": m["label"]} for k, m in p.MODELS.items()],
            "fields_url": f"/{p.PROJECT_ID}/fields",
            "predict_url": f"/{p.PROJECT_ID}/predict/{{model}}",
            "samples": p.SAMPLES,
        }
        for p in PROJECTS
    ]


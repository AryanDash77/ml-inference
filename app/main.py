import logging
from fastapi import FastAPI, HTTPException
from app.projects import attrition, fraud_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECTS = [fraud_detection, attrition]

app = FastAPI(title="ML Inference API", version="2.0.0",
              description="One server for several ML projects. Pick a project and a model, send named columns.")

for project in PROJECTS:
    app.include_router(project.router)
app.include_router(fraud_detection.legacy_router)  # old POST /predict


@app.get("/")
def root():
    return {"service": "ml-inference-api", "projects": [p.PROJECT_ID for p in PROJECTS]}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    missing = {p.PROJECT_ID: m for p in PROJECTS if (m := p.missing_files())}
    if missing:
        raise HTTPException(status_code=503, detail={"missing_files": missing})
    return {"status": "ready", "projects": [p.PROJECT_ID for p in PROJECTS]}


@app.get("/projects")
def list_projects():
    return [
        {
            "id": p.PROJECT_ID, "title": p.TITLE, "description": p.DESCRIPTION,
            "models": [{"id": k, "label": m["label"]} for k, m in p.MODELS.items()],
            "fields_url": f"/{p.PROJECT_ID}/fields",
            "predict_url": f"/{p.PROJECT_ID}/predict/{{model}}",
        }
        for p in PROJECTS
    ]



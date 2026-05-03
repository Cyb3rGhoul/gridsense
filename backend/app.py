from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline import DATA_DIR, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"

app = FastAPI(title="GridSense API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _load_json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        run_pipeline()
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "offline-demo", "sensitive_data": "none"}


@app.post("/api/pipeline/run")
def rebuild_pipeline():
    return run_pipeline(source="auto")


@app.get("/api/metrics")
def metrics():
    return _load_json("metrics.json")


@app.get("/api/forecasts")
def forecasts():
    return _load_json("forecasts.json")


@app.get("/api/anomalies")
def anomalies():
    return _load_json("anomalies.json")


@app.get("/api/zones")
def zones():
    return _load_json("zones.json")


@app.get("/api/anomaly-evidence")
def anomaly_evidence():
    return _load_json("anomaly_evidence.json")


@app.get("/api/pipeline")
def pipeline_summary():
    return _load_json("pipeline_summary.json")


@app.get("/api/theft-validation")
def theft_validation():
    return _load_json("theft_validation.json")


@app.get("/api/shap-explanations")
def shap_explanations():
    """Get SHAP explainability data for theft model."""
    shap_path = DATA_DIR / "shap_explanations.json"
    if shap_path.exists():
        return json.loads(shap_path.read_text(encoding="utf-8"))
    return {"available": False, "reason": "SHAP explanations not generated yet"}

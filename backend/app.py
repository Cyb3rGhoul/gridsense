from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.paths import DATA_DIR, ROOT, STATIC_DIR


PIPELINE_STATUS_PATH = DATA_DIR / "pipeline_status.json"
INSPECTION_FEEDBACK_PATH = DATA_DIR / "inspection_feedback.json"

app = FastAPI(title="GridSense API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_rebuild_lock = threading.Lock()
_artifact_lock = threading.Lock()
_rebuild_state: dict[str, object] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "last_result": None,
}
_ARTIFACT_FILES = {
    "metrics.json",
    "forecasts.json",
    "zones.json",
    "anomalies.json",
    "anomaly_evidence.json",
    "pipeline_summary.json",
    "theft_validation.json",
}


class InspectionFeedbackPayload(BaseModel):
    meter_id: str = Field(min_length=1)
    verdict: str = Field(min_length=1)
    feeder_id: str | None = None
    locality: str | None = None
    note: str | None = None


def _default_feedback_state() -> dict[str, object]:
    return {
        "records": [],
        "by_meter": {},
        "summary": {
            "total": 0,
            "confirmed_suspicious": 0,
            "false_alarm": 0,
            "cleared": 0,
            "latest_at": None,
        },
    }


def _load_feedback_state() -> dict[str, object]:
    if not INSPECTION_FEEDBACK_PATH.exists():
        return _default_feedback_state()
    try:
        loaded = json.loads(INSPECTION_FEEDBACK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_feedback_state()
    if not isinstance(loaded, dict):
        return _default_feedback_state()
    state = _default_feedback_state()
    state["records"] = loaded.get("records", [])
    state["by_meter"] = loaded.get("by_meter", {})
    state["summary"] = {**state["summary"], **loaded.get("summary", {})}
    return state


def _persist_feedback_state(state: dict[str, object]) -> None:
    INSPECTION_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSPECTION_FEEDBACK_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _normalize_feedback_verdict(verdict: str) -> str:
    normalized = verdict.strip().lower().replace(" ", "_")
    allowed = {"confirmed_suspicious", "false_alarm", "cleared"}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail="Invalid verdict")
    return normalized


def _persist_rebuild_state(state: dict[str, object]) -> None:
    PIPELINE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATUS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_persisted_rebuild_state() -> dict[str, object] | None:
    if not PIPELINE_STATUS_PATH.exists():
        return None
    try:
        loaded = json.loads(PIPELINE_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return {
        "status": loaded.get("status", "idle"),
        "started_at": loaded.get("started_at"),
        "finished_at": loaded.get("finished_at"),
        "last_error": loaded.get("last_error"),
        "last_result": loaded.get("last_result"),
    }


def _set_rebuild_state(**updates: object) -> None:
    with _rebuild_lock:
        _rebuild_state.update(updates)
        _persist_rebuild_state(_rebuild_state)


def _get_rebuild_state() -> dict[str, object]:
    with _rebuild_lock:
        return dict(_rebuild_state)


def _initialize_rebuild_state() -> None:
    persisted = _load_persisted_rebuild_state()
    if persisted is None:
        _persist_rebuild_state(_rebuild_state)
        return
    with _rebuild_lock:
        _rebuild_state.update(persisted)


def _run_pipeline_in_background(source: str = "auto") -> None:
    from backend.pipeline import run_pipeline

    try:
        result = run_pipeline(source=source)
        _set_rebuild_state(
            status="completed",
            finished_at=result.get("generated_at"),
            last_error=None,
            last_result=result,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime path
        _set_rebuild_state(
            status="failed",
            finished_at=datetime.now(UTC).isoformat(),
            last_error=str(exc),
        )


def _load_json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        with _artifact_lock:
            if not path.exists():
                from backend.pipeline import run_pipeline

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
    state = _get_rebuild_state()
    if state["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline rebuild already running")

    started_at = datetime.now(UTC).isoformat()
    _set_rebuild_state(
        status="running",
        started_at=started_at,
        finished_at=None,
        last_error=None,
    )
    worker = threading.Thread(
        target=_run_pipeline_in_background,
        kwargs={"source": "auto"},
        daemon=True,
        name="gridsense-pipeline-rebuild",
    )
    worker.start()
    return {
        "status": "accepted",
        "message": "Pipeline rebuild started in background",
        "started_at": started_at,
    }


@app.get("/api/pipeline/status")
def pipeline_status():
    return _get_rebuild_state()


@app.get("/api/inspection-feedback")
def inspection_feedback():
    return _load_feedback_state()


@app.post("/api/inspection-feedback")
def submit_inspection_feedback(payload: InspectionFeedbackPayload):
    verdict = _normalize_feedback_verdict(payload.verdict)
    feedback = _load_feedback_state()
    now = datetime.now(UTC).isoformat()
    record = {
        "meter_id": payload.meter_id,
        "verdict": verdict,
        "feeder_id": payload.feeder_id,
        "locality": payload.locality,
        "note": payload.note,
        "recorded_at": now,
    }
    records = feedback["records"]
    if not isinstance(records, list):
        records = []
    records.append(record)
    feedback["records"] = records[-200:]
    by_meter = feedback.get("by_meter", {})
    if not isinstance(by_meter, dict):
        by_meter = {}
    by_meter[payload.meter_id] = record
    feedback["by_meter"] = by_meter

    summary = feedback.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    recent_records = feedback["records"]
    summary.update(
        total=len(recent_records),
        confirmed_suspicious=sum(1 for item in recent_records if item.get("verdict") == "confirmed_suspicious"),
        false_alarm=sum(1 for item in recent_records if item.get("verdict") == "false_alarm"),
        cleared=sum(1 for item in recent_records if item.get("verdict") == "cleared"),
        latest_at=now,
    )
    feedback["summary"] = summary
    _persist_feedback_state(feedback)
    return {"status": "recorded", "record": record, "summary": summary}


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


_initialize_rebuild_state()

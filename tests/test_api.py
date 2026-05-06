from fastapi.testclient import TestClient
from pathlib import Path
import json
import threading
import time

import backend.app as app_module
import backend.pipeline as pipeline_module


class DummyThread:
    def __init__(self, target=None, kwargs=None, daemon=None, name=None):
        self.target = target
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self):
        self.started = True


client = TestClient(app_module.app)
TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def setup_function():
    TMP_ROOT.mkdir(exist_ok=True)
    for path in sorted(TMP_ROOT.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    app_module._set_rebuild_state(
        status="idle",
        started_at=None,
        finished_at=None,
        last_error=None,
        last_result=None,
    )
    if app_module.PIPELINE_STATUS_PATH.exists():
        app_module.PIPELINE_STATUS_PATH.unlink()
    if app_module.INSPECTION_FEEDBACK_PATH.exists():
        app_module.INSPECTION_FEEDBACK_PATH.unlink()


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_pipeline_run_is_non_blocking(monkeypatch):
    monkeypatch.setattr(app_module.threading, "Thread", DummyThread)

    response = client.post("/api/pipeline/run")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["started_at"]

    status = client.get("/api/pipeline/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "running"
    assert payload["started_at"] == body["started_at"]


def test_pipeline_run_rejects_duplicate_request():
    app_module._set_rebuild_state(status="running", started_at="2026-05-04T00:00:00+00:00")

    response = client.post("/api/pipeline/run")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_rebuild_state_is_persisted_to_disk(monkeypatch):
    status_path = TMP_ROOT / "pipeline_status.json"
    monkeypatch.setattr(app_module, "PIPELINE_STATUS_PATH", status_path)

    app_module._set_rebuild_state(
        status="completed",
        started_at="2026-05-04T00:00:00+00:00",
        finished_at="2026-05-04T00:01:00+00:00",
        last_error=None,
        last_result={"ok": True},
    )

    assert status_path.exists()
    persisted = app_module._load_persisted_rebuild_state()
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["last_result"] == {"ok": True}


def test_initialize_rebuild_state_reads_persisted_file(monkeypatch):
    status_path = TMP_ROOT / "pipeline_status.json"
    status_path.write_text(
        '{"status":"failed","started_at":"2026-05-04T00:00:00+00:00","finished_at":"2026-05-04T00:02:00+00:00","last_error":"boom","last_result":null}',
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "PIPELINE_STATUS_PATH", status_path)

    app_module._rebuild_state.update(
        status="idle",
        started_at=None,
        finished_at=None,
        last_error=None,
        last_result=None,
    )
    app_module._initialize_rebuild_state()

    state = app_module._get_rebuild_state()
    assert state["status"] == "failed"
    assert state["last_error"] == "boom"


def test_inspection_feedback_is_persisted(monkeypatch):
    feedback_path = TMP_ROOT / "inspection_feedback.json"
    monkeypatch.setattr(app_module, "INSPECTION_FEEDBACK_PATH", feedback_path)

    response = client.post(
        "/api/inspection-feedback",
        json={
            "meter_id": "MAC001",
            "verdict": "false_alarm",
            "feeder_id": "F01",
            "locality": "Indiranagar",
            "note": "Bench verified",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "recorded"
    assert body["summary"]["false_alarm"] == 1
    assert feedback_path.exists()

    stored = client.get("/api/inspection-feedback")
    assert stored.status_code == 200
    payload = stored.json()
    assert payload["summary"]["total"] == 1
    assert payload["by_meter"]["MAC001"]["verdict"] == "false_alarm"


def test_missing_artifact_triggers_single_pipeline_run(monkeypatch):
    data_dir = TMP_ROOT / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "DATA_DIR", data_dir)

    calls = {"count": 0}
    lock = threading.Lock()

    def fake_run_pipeline(source="auto"):
        with lock:
            calls["count"] += 1
        time.sleep(0.1)
        (data_dir / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return {"status": "ok"}

    monkeypatch.setattr(pipeline_module, "run_pipeline", fake_run_pipeline)

    results = []

    def worker():
        results.append(app_module._load_json("metrics.json"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["count"] == 1
    assert results == [{"status": "ok"}] * 4

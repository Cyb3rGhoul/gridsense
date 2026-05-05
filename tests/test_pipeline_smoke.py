import json
from pathlib import Path

import backend.pipeline as pipeline_module


TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def test_run_pipeline_rebuilds_artifacts_in_isolated_dir(monkeypatch):
    root = TMP_ROOT / "pipeline_smoke"
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        root.rmdir()
    data_dir = root / "processed"
    models_dir = root / "models"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    monkeypatch.setattr(pipeline_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(pipeline_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(pipeline_module, "SGCC_METRICS_JSON", data_dir / "theft_validation.json")
    cached_validation = {
        "available": True,
        "f1": 0.5,
        "pr_auc": 0.5,
        "roc_auc": 0.8,
        "feature_importance": [{"feature": "recent_drop_ratio", "importance": 0.4}],
        "top_cases": [{"consumer_id": "demo", "label": 1, "theft_probability": 0.8, "recent_drop_pct": 30.0, "missing_rate": 0.0, "zero_rate": 0.1, "explanation": "demo"}],
    }
    monkeypatch.setattr(pipeline_module, "load_cached_sgcc_theft_validation", lambda: cached_validation)

    metrics = pipeline_module.run_pipeline(source="synthetic")

    assert metrics["feeders"] == 10
    assert metrics["meters"] >= 50

    expected_outputs = [
        "metrics.json",
        "forecasts.json",
        "anomalies.json",
        "zones.json",
        "anomaly_evidence.json",
        "pipeline_summary.json",
        "theft_validation.json",
        "sample_meter_readings.csv",
    ]
    for name in expected_outputs:
        path = data_dir / name
        assert path.exists()
        assert path.stat().st_size > 0

    zone_rows = json.loads((data_dir / "zones.json").read_text(encoding="utf-8"))
    assert zone_rows
    assert all("dispatch_score" in row for row in zone_rows)
    assert all("zone_priority" in row for row in zone_rows)

    anomaly_rows = json.loads((data_dir / "anomalies.json").read_text(encoding="utf-8"))
    assert anomaly_rows
    assert all("inspection_priority" in row for row in anomaly_rows)
    assert all("false_positive_risk" in row for row in anomaly_rows)

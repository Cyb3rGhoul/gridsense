import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_processed_metrics_are_current_shape():
    metrics = json.loads((ROOT / "data" / "processed" / "metrics.json").read_text())
    assert metrics["meters"] >= 8
    assert metrics["feeders"] >= 8
    assert "forecast_smape" in metrics
    assert metrics["sgcc_theft_validation"] is True


def test_anomalies_have_meaningful_signals():
    anomalies = json.loads((ROOT / "data" / "processed" / "anomalies.json").read_text())
    assert anomalies
    # At least some anomalies should have sub-1.0 drop ratios
    assert any(item["drop_ratio"] < 0.95 for item in anomalies)


def test_theft_validation_quality():
    theft = json.loads((ROOT / "data" / "processed" / "theft_validation.json").read_text())
    assert theft["available"] is True
    assert theft["f1"] > 0.4
    assert theft["roc_auc"] > 0.75
    assert "confusion_matrix" in theft
    assert "roc_curve" in theft
    assert "pr_curve" in theft
    assert len(theft["feature_importance"]) >= 10
    assert len(theft["top_cases"]) >= 10

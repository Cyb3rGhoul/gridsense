# GridSense Submission

## Refined Positioning

GridSense should be pitched as a utility engineer cockpit, not as another ML model. The winning angle is: BESCOM already has smart-meter data; GridSense makes it operationally useful without modifying any existing system.

## Sharpened Differentiators

- **Inspection ROI scoring:** anomaly flags are ranked by confidence and estimated revenue impact, so field teams act on the highest-value cases first.
- **False-positive discipline:** confidence tiers are visible, and the dashboard is designed to incorporate inspection outcomes as feedback.
- **Zone-level operations:** the product forecasts at feeder/locality level, then combines peak risk and theft risk into a single dispatch priority.
- **Auditability:** every generated alert has feature evidence: baseline drop, peer deviation, capacity proximity, and time window.
- **On-premise ready:** no hosted LLMs; all models can run inside BESCOM infrastructure, and a Dockerfile is included for deployment.

## Architecture

1. Data ingestion reads smart-meter exports or stream replicas.
2. Feature jobs clean gaps, build weather/calendar features, aggregate meters to feeders, and construct peer groups.
3. LightGBM forecasting model predicts 4-hour and 24-hour feeder demand.
4. Operational anomaly engine combines individual baseline deviation, peer comparison, and unsupervised outlier scoring.
5. SGCC theft-validation lab trains a leakage-free 5-fold OOF stacked ensemble (LightGBM + XGBoost + ExtraTrees + HistGradientBoosting + CatBoost) on labelled theft data, with BorderlineSMOTE per fold, prototype-margin meta-features, ensemble label-noise correction, and an isotonic-calibrated logistic-regression meta-learner. Threshold is picked on out-of-fold predictions only, never on the test set.
6. Decision API serves forecasts, zone priorities, anomaly queue, model validation metrics, explanations, and audit metadata.
7. Dashboard gives engineers interactive forecasts, risk zones, anomaly evidence, confusion matrix, ROC/PR curves, and exportable theft cases.

## Evaluation Plan

- Forecasting: compare sMAPE/MAE against a 24-hour lag persistence baseline.
- Theft detection: measure precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and top-k case quality on SGCC.
- Operations: measure top-k inspection yield and recovered/recoverable revenue per inspection.
- Robustness: test missing data, holidays, seasonal heat load, low-consumption users, and commercial weekend closures.

## Demo Script

1. Show dashboard: "This is a read-only intelligence layer on top of BESCOM smart-meter exports."
2. Open the forecast panel: "The feeder forecast shows the next 24 hours with uncertainty bands and stress windows."
3. Open risk zones: "Zones are prioritized by peak load proximity and unresolved high-confidence anomalies."
4. Open anomaly queue: "Each meter flag has confidence, estimated revenue risk, peer/baseline evidence, and recommended action."
5. Open SGCC Theft Detection Lab: "This validates theft classification on real labelled theft data using an ensemble, not on unlabelled operational data."
6. Rebuild the pipeline live only if time permits: "The full rebuild retrains models and refreshes outputs; the dashboard otherwise loads persisted artifacts instantly."

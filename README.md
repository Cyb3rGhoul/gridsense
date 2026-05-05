# GridSense

AI-powered smart meter intelligence and loss-detection workflow for BESCOM Theme 8.

GridSense is a decision-support layer. It does not modify utility systems, does not use hosted LLMs on sensitive data, and is built to turn meter data into feeder risk, inspection priority, and explainable evidence.

## What It Does

### Part A - Localized Demand Prediction
- Forecasts short-term feeder demand
- Surfaces peak-load windows and locality-level grid stress
- Ranks feeders and localities into Normal, High, and Critical dispatch zones

### Part B - Anomaly and Loss Detection
- Scores abnormal consumption behavior at meter level
- Separates likely variability from suspicious deviation using baseline, peer, and persistence signals
- Produces an inspection-ready queue with confidence, false-positive controls, and evidence

## Current Data Architecture

GridSense now uses a split operational stack instead of one blended demo source.

- Forecast branch:
  `Official SSEN LV feeder smart-meter aggregated demand export`
- Anomaly branch:
  `Real London smart-meter usage data with weather joins, scored in operational unsupervised mode`
- Separate validation lab:
  `SGCC labelled theft benchmark`

This separation is intentional. The live dashboard does not pretend that an external labelled benchmark is the same thing as live utility operations.

## Pipeline Summary

The orchestration entrypoint is [backend/pipeline.py](backend/pipeline.py).

1. Forecast pipeline
   - loads feeder demand data
   - engineers time, lag, rolling, and weather features
   - trains a LightGBM regressor
   - generates forward forecasts, uncertainty bands, and feeder risk

2. Anomaly pipeline
   - loads meter-level behavioral data
   - builds baseline-drop, peer-ratio, volatility, and persistence features
- scores suspicious meters in operational unsupervised mode
   - applies queue thresholds and false-positive controls
   - generates evidence payloads and inspection actions

3. Zone pipeline
   - merges forecast stress with anomaly exposure
   - computes dispatch scores
   - outputs zone priorities and actions

4. Serving layer
   - FastAPI serves JSON artifacts and the dashboard
   - pipeline rebuilds run in the background
   - rebuild state is persisted
   - inspection feedback is persisted

## Dashboard Structure

The UI is organized into operator views instead of one long report.

- `Overview`
- `Forecast`
- `Risk Map`
- `Queue`
- `Validation Lab`
- `Pipeline`

The dashboard includes:
- a real Leaflet-based risk map
- feeder forecast charts
- an operational anomaly queue
- an evidence panel with review actions
- a separate SGCC validation lab

## False-Positive Handling

This project does not claim perfect theft detection. It is built as an inspection-prioritization layer.

Visible false-positive controls include:
- precision-biased queue thresholds
- peer-support and persistence gating
- evidence-first explanations
- persisted field review outcomes:
  - `confirmed_suspicious`
  - `false_alarm`
  - `cleared`

Field reviews are available through the evidence panel and are stored by the API for auditability and threshold refinement.

## API Endpoints

- `GET /api/health`
- `GET /api/metrics`
- `GET /api/forecasts`
- `GET /api/zones`
- `GET /api/anomalies`
- `GET /api/anomaly-evidence`
- `GET /api/pipeline`
- `GET /api/theft-validation`
- `GET /api/pipeline/status`
- `GET /api/inspection-feedback`
- `POST /api/pipeline/run`
- `POST /api/inspection-feedback`

## Local Run

```bash
pip install -r requirements.txt
python -m backend.pipeline
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Deployment Split

For a stable demo link, deploy the backend and frontend separately:

1. Render for the API
   - use `render.yaml`
   - this serves the FastAPI app and JSON endpoints

2. Vercel for the dashboard
   - Vercel runs `npm run build`
   - that generates `public/` and writes `public/static/config.js`
   - set Vercel env var `GRIDSENSE_API_BASE` to your Render URL, for example:
     `https://gridsense.onrender.com`

This makes the Vercel page call the Render API instead of looking for `/api/*` on Vercel itself.

### Vercel Notes

- `vercel.json` points Vercel at the generated `public/` directory
- `.vercelignore` excludes raw datasets and backend code from the frontend deploy bundle
- if `GRIDSENSE_API_BASE` is empty, the frontend falls back to same-origin API calls

## Tests

```bash
python -m pytest tests -q
```

Current coverage includes:
- artifact output checks
- API behavior checks
- rebuild status persistence
- inspection feedback persistence
- frontend asset checks
- pipeline smoke tests

## Project Positioning For BESCOM

GridSense is strongest when pitched as:

- an operational intelligence layer, not a black-box verdict engine
- a combined feeder-risk and inspection-priority workflow
- an explainable and auditable smart-meter decision-support product
- a system that can improve over time through field feedback

---
title: LC8 Credit Scoring API
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# LC8 Credit Scoring API

FastAPI credit scoring service that predicts a customer's default risk using a
LightGBM model trained on Home Credit data.

The API exposes the top 20 SHAP features. The remaining model features are
automatically filled with the training medians before inference.

## Features

- FastAPI service with automatic Swagger documentation on `/docs`.
- Token-protected `/predict` endpoint using Bearer authentication.
- Input validation with Pydantic.
- Model loaded once at application startup and reused across requests.
- Optimized ONNX Runtime inference with LightGBM fallback.
- Automated tests with `pytest`.
- Dockerized API runtime.
- GitHub Actions pipeline for testing and deployment to Hugging Face Spaces.
- Real-time data drift monitoring with batch-based detection.
- Structured logging of predictions (inputs, outputs, latency).
- Optional PostgreSQL/Supabase persistence for prediction and drift monitoring.

## Monitoring & Drift Detection

The API includes a monitoring system to track both data drift and operational performance.

### Logging

Each prediction request is logged as structured JSON containing:
- input features (top SHAP features)
- model outputs (prediction, probability, risk category)
- inference time (latency)

Logs are stored locally in rotating files:
```text
logs/predictions.log
```

Prediction and drift events can also be persisted to a database for analytics.
The database layer is optional: if no monitoring database environment variables
are configured, the API keeps running with file/console logging only.

Two tables are used:

```text
prediction_logs
drift_events
```

The schema is available in:

```text
monitoring/schema.sql
```

### Drift Detection

A drift monitoring component is integrated into the API:

- predictions are buffered in memory;
- drift analysis is triggered every N requests (batch size, default = 100);
- production data is compared to reference (training) data.

Drift detection uses:
- Evidently (if available)
- Kolmogorov–Smirnov test (fallback)

### Drift Logs

When drift analysis runs, logs are generated:

```json
{
  "event": "drift_analysis_completed",
  "drift_detected": true,
  "drifted_features": 2
}
```

If drift is detected at feature level:

```json
{
  "event": "feature_drift_detected",
  "feature": "AMT_CREDIT"
}
```

### Operational Anomaly Analysis

The persisted `prediction_logs` table can be analyzed automatically for
operational issues:

- error rate;
- average latency;
- p95 latency;
- maximum latency.

Run the analysis locally after generating a few prediction records:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://mlops:mlops@localhost:5432/monitoring"
uv run python -m monitoring.analyze_operational_metrics
```

The command outputs a JSON report with status `ok` or `alert`. Thresholds can be
configured:

```powershell
uv run python -m monitoring.analyze_operational_metrics `
  --error-rate-threshold 0.05 `
  --latency-p95-threshold-ms 1000 `
  --lookback-hours 24
```

Use `--fail-on-alert` if the command should return a non-zero exit code when an
alert is detected.

### Purpose

This monitoring system allows:

- early detection of data distribution shifts;
- identification of impacted features;
- detection of abnormal error rate or prediction latency;
- triggering of corrective actions (performance check, retraining).

In production, these logs can be stored in PostgreSQL/Supabase and integrated
into monitoring tools such as ELK, Grafana, or dashboard notebooks.

### Validation

The monitoring system is validated through automated tests that simulate API
calls, verify that drift detection is triggered once the batch size threshold is
reached, and check operational anomaly detection logic.

## Architecture

The system follows a production-oriented MLOps design:

Client → API → Model
↓
Logging → Drift Monitor

- The API serves predictions using a preloaded model.
- Each request is logged (inputs, outputs, latency).
- Prediction and drift records can be persisted to PostgreSQL/Supabase.
- A drift monitor collects predictions and performs periodic analysis.
- Drift detection results are logged for monitoring and alerting.
- An operational analysis script detects high error rate and abnormal latency.

This design simulates a real-world ML monitoring pipeline.

## Performance Optimization

The inference path was optimized after deployment using monitoring data,
benchmarking, and `cProfile`.

The first profiling pass showed that the original LightGBM path spent a
significant part of inference time converting pandas `DataFrame` inputs into
LightGBM's internal format. The API was first optimized to build ordered NumPy
arrays directly from validated Pydantic inputs.

ONNX Runtime was then evaluated as a second optimization step. The LightGBM
model was exported to ONNX with `target_opset=15` and benchmarked against the
optimized LightGBM NumPy path.

```text
LightGBM NumPy mean latency: 0.9951 ms
ONNX Runtime mean latency:   0.0305 ms
Model hot-path improvement:  96.94%
Max probability difference:  0.000000131269
```

ONNX Runtime is now the default inference runtime. LightGBM remains available as
a fallback with `MODEL_RUNTIME=lightgbm`. The reported improvement applies to
the model hot path; complete HTTP latency also includes FastAPI, validation,
logging, drift monitoring, serialization, and network overhead.

The detailed optimization report is available in:

```text
docs/performance_optimization_report.md
```


## Requirements

- Python 3.13
- `uv`
- Docker Desktop
- Git
- A Hugging Face Space configured with Docker

## Local Setup

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync --frozen
```

## Run Tests

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv run pytest tests/ -v
```

The test suite covers:

- API contract and endpoints;
- authentication and error handling;
- model inference and feature consistency;
- drift monitoring trigger (batch-based detection);
- logging behavior.

## Run the API Locally

Optional: start local PostgreSQL first if you want database persistence.

```powershell
docker compose up -d postgres
```

Then set the monitoring database URL:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://mlops:mlops@localhost:5432/monitoring"
```

The API auto-creates the monitoring tables by default. To manage the schema
yourself, run `monitoring/schema.sql` manually and set:

```powershell
$env:MONITORING_AUTO_CREATE_TABLES = "false"
```

Terminal 1:

```powershell
$env:API_TOKEN = "test-token"
uv run uvicorn app.main:app --host 127.0.0.1 --port 7860
```

Terminal 2:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
```

```powershell
Invoke-RestMethod http://127.0.0.1:7860/features
```

```powershell
Invoke-RestMethod `
  http://127.0.0.1:7860/predict `
  -Method Post `
  -Headers @{ Authorization = "Bearer test-token" } `
  -ContentType "application/json" `
  -Body "{}"
```

Interactive documentation is available at:

```text
http://127.0.0.1:7860/docs
```

The API uses ONNX Runtime by default for optimized inference. To force the
LightGBM runtime locally:

```powershell
$env:MODEL_RUNTIME = "lightgbm"
uv run uvicorn app.main:app --host 127.0.0.1 --port 7860
```

To switch back to the default ONNX runtime:

```powershell
Remove-Item Env:MODEL_RUNTIME
```

To inspect persisted predictions locally:

```powershell
docker compose exec postgres psql -U mlops -d monitoring -c "select id, timestamp, prediction, probability_of_default, risk_category from prediction_logs order by id desc limit 5;"
```

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/health` | Checks that the API is running and the model is loaded. |
| `GET` | `/features` | Returns the exposed input features expected by the API. |
| `POST` | `/predict` | Returns the prediction, default probability, and risk category. |

## Public Deployment

Hugging Face Space:

```text
https://huggingface.co/spaces/professor-chen/LC_8
```

Public Swagger documentation:

```text
https://professor-chen-lc-8.hf.space/docs
```

Public health check:

```text
https://professor-chen-lc-8.hf.space/health
```

## Docker

Build the image:

```powershell
docker build -t lc8-credit-scoring-api:local .
```

Run the container:

```powershell
docker run --rm -d `
  -p 7860:7860 `
  -e API_TOKEN=test-token `
  --name lc8-api `
  lc8-credit-scoring-api:local
```

Test the container:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
```

Stop the container:

```powershell
docker stop lc8-api
```

## CI/CD

The GitHub Actions pipeline is defined in:

```text
.github/workflows/ci-cd.yml
```

It runs the following steps:

- installs Python 3.13 and dependencies with `uv`;
- runs the automated test suite with `pytest`;
- deploys to Hugging Face Spaces after successful tests;
- handles binary files for the Space with Git LFS / Xet.

Deployment to Hugging Face is triggered by a push to `main`. The Hugging Face
Space then uses the `Dockerfile` to build and run the containerized API.

## Required Secrets

GitHub Actions secrets:

```text
HF_TOKEN
HF_USER_SPACE_NAME
```

`HF_USER_SPACE_NAME` must use this format:

```text
username/space-name
```

Hugging Face Space secret:

```text
API_TOKEN
```

Optional monitoring secrets for Supabase over HTTPS:

```text
MONITORING_BACKEND=supabase
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

Before enabling these, run `monitoring/schema.sql` in the Supabase SQL editor.
This path uses Supabase's HTTPS REST API, which is safer for Hugging Face Spaces
than assuming a direct PostgreSQL TCP connection is available.

This token is required by the `/predict` endpoint through the header:

```text
Authorization: Bearer <API_TOKEN>
```

## Useful Environment Variables

| Variable | Description |
| --- | --- |
| `API_TOKEN` | Required token for calling `/predict`. |
| `LOCAL_MODEL_URI` | Local path to the MLflow model artifact. |
| `MLFLOW_TRACKING_URI` | Optional MLflow URI for loading the model from a registry. |
| `MODEL_NAME` | MLflow model name when using the registry. |
| `MODEL_VERSION` | MLflow model version when using the registry. |
| `MODEL_RUNTIME` | Inference runtime: `onnx` or `lightgbm`. Defaults to `onnx`. |
| `ONNX_MODEL_PATH` | Optional path to the ONNX model. Defaults to `models/onnx/home_credit_lightgbm.onnx`. |
| `DATABASE_URL` | SQLAlchemy URL for local PostgreSQL or a reachable Postgres database. |
| `MONITORING_BACKEND` | `auto`, `supabase`, or `disabled`. Defaults to `auto`. |
| `MONITORING_AUTO_CREATE_TABLES` | Auto-create SQLAlchemy tables when `DATABASE_URL` is set. Defaults to `true`. |
| `SUPABASE_URL` | Supabase project URL for HTTPS monitoring writes. |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-side Supabase key for inserting monitoring records. |
| `SUPABASE_PREDICTION_TABLE` | Optional table override. Defaults to `prediction_logs`. |
| `SUPABASE_DRIFT_TABLE` | Optional table override. Defaults to `drift_events`. |
| `OPERATIONAL_LOOKBACK_HOURS` | Lookback window for operational analysis. Defaults to `24`. |
| `OPERATIONAL_ERROR_RATE_THRESHOLD` | Error-rate alert threshold. Defaults to `0.05`. |
| `OPERATIONAL_LATENCY_P95_THRESHOLD_MS` | p95 latency alert threshold in ms. Defaults to `1000`. |
| `OPERATIONAL_MIN_REQUESTS` | Minimum request count before error-rate alerting. Defaults to `1`. |

## Versioning Notes

Local caches are not required to run the project and should not be deployed:

```text
.pytest_cache/
.uv-cache/
__pycache__/
.venv/
```

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
- Automated tests with `pytest`.
- Dockerized API runtime.
- GitHub Actions pipeline for testing and deployment to Hugging Face Spaces.

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

- the API contract;
- model loading;
- model feature consistency;
- input validation;
- authentication errors.

## Run the API Locally

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

## Versioning Notes

Local caches are not required to run the project and should not be deployed:

```text
.pytest_cache/
.uv-cache/
__pycache__/
.venv/
```

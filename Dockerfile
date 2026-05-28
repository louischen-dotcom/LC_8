FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PORT=7860 \
    LOCAL_MODEL_URI=/app/mlflow/artifacts/models/m-09098a8dc27c4199bc934fafb610fdb1/artifacts

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version README.md ./

RUN python -m pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY monitoring ./monitoring
COPY mlflow/artifacts/models ./mlflow/artifacts/models

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

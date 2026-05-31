import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import onnxruntime as ort


_model: Any | None = None


def get_model_runtime() -> str:
    return os.environ.get("MODEL_RUNTIME", "onnx").lower()


def _default_local_model_path() -> Path:
    artifact_root = Path.cwd() / "mlflow" / "artifacts" / "models"
    candidates = sorted(artifact_root.glob("*/artifacts/MLmodel"))
    if not candidates:
        raise FileNotFoundError(f"No local MLflow model found under {artifact_root}")
    return candidates[0].parent


def _default_onnx_model_path() -> Path:
    return Path.cwd() / "models" / "onnx" / "home_credit_lightgbm.onnx"


def load_onnx_model() -> ort.InferenceSession:
    onnx_model_path = Path(
        os.environ.get("ONNX_MODEL_PATH", str(_default_onnx_model_path()))
    )

    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    model = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
    )
    print(f"Model loaded from ONNX Runtime: {onnx_model_path}")
    return model


def load_lightgbm_model() -> Any:
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{Path.cwd() / 'mlflow' / 'mlflow.db'}",
    )
    model_name = os.environ.get("MODEL_NAME", "home_credit_scoring")
    model_version = os.environ.get("MODEL_VERSION", "1")
    local_model_uri = os.environ.get("LOCAL_MODEL_URI")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    registry_model_uri = f"models:/{model_name}/{model_version}"

    if local_model_uri:
        try:
            model = mlflow.lightgbm.load_model(local_model_uri)
            print(f"Model loaded from local MLflow artifact: {local_model_uri}")
            return model
        except Exception as local_error:
            raise RuntimeError(
                f"Failed to load model from LOCAL_MODEL_URI '{local_model_uri}'. "
                f"Error: {local_error}"
            ) from local_error

    try:
        model = mlflow.lightgbm.load_model(registry_model_uri)
        print(f"Model loaded from MLflow Registry: {registry_model_uri}")
        return model
    except Exception as registry_error:
        fallback_path = _default_local_model_path()
        try:
            model = mlflow.lightgbm.load_model(str(fallback_path))
            print(f"Model loaded from local MLflow artifact: {fallback_path}")
            return model
        except Exception as local_error:
            raise RuntimeError(
                f"Failed to load model from registry '{registry_model_uri}' "
                f"or local artifact '{fallback_path}'. "
                f"Registry error: {registry_error}. Local error: {local_error}"
            ) from local_error


def load_model() -> Any:
    """Load the configured model runtime once."""
    global _model

    if _model is not None:
        return _model

    runtime = get_model_runtime()

    if runtime == "onnx":
        _model = load_onnx_model()
        return _model

    if runtime == "lightgbm":
        _model = load_lightgbm_model()
        return _model

    raise ValueError(
        f"Unsupported MODEL_RUNTIME '{runtime}'. "
        "Expected 'onnx' or 'lightgbm'."
    )


def get_model() -> Any:
    """Get the loaded model."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model
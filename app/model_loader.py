# app/model_loader.py

import joblib
import os
_model = None
def load_model():
    """Load the model ONCE at startup."""
    global _model

    # Allow the path to be configured via environment variable
    model_path = os.environ.get("MODEL_PATH", "model/credit_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run train_model.py first."
        )

    _model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return _model
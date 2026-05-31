import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import build_model_frame, predict_default_probability
from app.model_loader import load_model
from app.schemas import CreditApplication, MODEL_FEATURES


def predict_numpy(model, application):
    model_input = application.to_model_input()
    model_array = np.array(
        [[model_input[feature] for feature in MODEL_FEATURES]],
        dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        probabilities = model.predict_proba(model_array)

    probability = float(probabilities[0][1])
    prediction = int(probability >= 0.5)
    return prediction, probability


def main():
    model = load_model()

    cases = [
        CreditApplication(),
        CreditApplication(AMT_CREDIT=300000, AMT_ANNUITY=18000),
        CreditApplication(AMT_CREDIT=900000, AMT_GOODS_PRICE=850000),
        CreditApplication(DAYS_BIRTH=-10000, DAYS_ID_PUBLISH=-2000),
        CreditApplication(
            EXT_SOURCE_MEAN=0.35,
            EXT_SOURCE_MIN=0.2,
            EXT_SOURCE_3=0.4,
        ),
    ]

    max_diff = 0.0

    for index, application in enumerate(cases, start=1):
        model_frame = build_model_frame(application)
        df_prediction, df_probability = predict_default_probability(model, model_frame)
        np_prediction, np_probability = predict_numpy(model, application)

        diff = abs(df_probability - np_probability)
        max_diff = max(max_diff, diff)

        print(f"Case {index}")
        print(f"  DataFrame: prediction={df_prediction}, probability={df_probability:.10f}")
        print(f"  NumPy:     prediction={np_prediction}, probability={np_probability:.10f}")
        print(f"  Diff:      {diff:.12f}")

        if df_prediction != np_prediction:
            raise RuntimeError(f"Prediction mismatch on case {index}")

        if diff > 1e-8:
            raise RuntimeError(f"Probability mismatch on case {index}: {diff}")

    print("")
    print(f"All cases matched. Max probability diff: {max_diff:.12f}")


if __name__ == "__main__":
    main()
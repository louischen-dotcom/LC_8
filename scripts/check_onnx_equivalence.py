import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import build_model_array, predict_default_probability
from app.model_loader import load_model
from app.schemas import CreditApplication
import os
import app.model_loader as model_loader


def predict_onnx(session, input_name, output_names, application):
    model_array = build_model_array(application).astype(np.float32)
    outputs = session.run(output_names, {input_name: model_array})

    prediction = int(outputs[0][0])
    probability = float(outputs[1][0][1])

    return prediction, probability


def main():
    os.environ["MODEL_RUNTIME"] = "lightgbm"
    model_loader._model = None
    model = load_model()

    onnx_path = PROJECT_ROOT / "models" / "onnx" / "home_credit_lightgbm.onnx"
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

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
    tolerance = 1e-5

    for index, application in enumerate(cases, start=1):
        model_array = build_model_array(application)
        lgb_prediction, lgb_probability = predict_default_probability(
            model,
            model_array,
        )
        onnx_prediction, onnx_probability = predict_onnx(
            session,
            input_name,
            output_names,
            application,
        )

        diff = abs(lgb_probability - onnx_probability)
        max_diff = max(max_diff, diff)

        print(f"Case {index}")
        print(
            f"  LightGBM: prediction={lgb_prediction}, "
            f"probability={lgb_probability:.10f}"
        )
        print(
            f"  ONNX:     prediction={onnx_prediction}, "
            f"probability={onnx_probability:.10f}"
        )
        print(f"  Diff:     {diff:.12f}")

        if lgb_prediction != onnx_prediction:
            raise RuntimeError(f"Prediction mismatch on case {index}")

        if diff > tolerance:
            raise RuntimeError(
                f"Probability mismatch on case {index}: "
                f"{diff} > {tolerance}"
            )

    print("")
    print(f"All cases matched. Max probability diff: {max_diff:.12f}")
    print(f"Tolerance: {tolerance}")


if __name__ == "__main__":
    main()
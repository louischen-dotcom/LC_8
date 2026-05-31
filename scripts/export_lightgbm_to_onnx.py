import sys
from pathlib import Path

from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_loader import load_model
from app.schemas import MODEL_FEATURES


def main():
    model = load_model()

    output_dir = PROJECT_ROOT / "models" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "home_credit_lightgbm.onnx"

    initial_types = [
        ("input", FloatTensorType([None, len(MODEL_FEATURES)]))
    ]

    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_types,
        target_opset=15,
        zipmap=False,
    )

    output_path.write_bytes(onnx_model.SerializeToString())

    print(f"Exported ONNX model to: {output_path}")
    print(f"Input shape: [None, {len(MODEL_FEATURES)}]")


if __name__ == "__main__":
    main()
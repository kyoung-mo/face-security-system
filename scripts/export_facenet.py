import torch
from facenet_pytorch import InceptionResnetV1
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_path = BASE_DIR / "models" / "facenet.onnx"

print("[INFO] FaceNet 모델 로드 중...")
model = InceptionResnetV1(pretrained='vggface2').eval()

dummy_input = torch.randn(1, 3, 160, 160)

print("[INFO] ONNX export 중...")
torch.onnx.export(
    model,
    dummy_input,
    str(output_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)

print(f"[INFO] 완료: {output_path}")

# 검증
import onnx
model_onnx = onnx.load(str(output_path))
onnx.checker.check_model(model_onnx)
print("[INFO] ONNX 모델 검증 완료")

from ultralytics import YOLO
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "yolov8n-face.pt"
output_dir = BASE_DIR / "models"

print("[INFO] YOLOv8n-face ONNX export 중...")
model = YOLO(str(model_path))

model.export(
    format="onnx",
    imgsz=320,
    opset=11,
    dynamic=False,
)

exported = BASE_DIR / "models" / "yolov8n-face.onnx"
target = output_dir / "yolov8_face.onnx"
shutil.move(str(exported), str(target))
print(f"[INFO] 완료: {target}")

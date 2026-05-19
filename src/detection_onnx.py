from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "yolov8_face.onnx"


class Detector:
    def __init__(self, model_path=None, conf_threshold: float = 0.2):
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_absolute():
            path = BASE_DIR / path
        self.model_path = str(path)
        self.conf_threshold = conf_threshold
        print(f"[Detector ONNX] Loading: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

    def detect(self, frame):
        results = self.model.predict(
            source=frame, imgsz=320,
            conf=self.conf_threshold, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        dets  = [(int(x1), int(y1), int(x2), int(y2), float(c))
                 for (x1,y1,x2,y2), c in zip(xyxy, confs)]
        if not dets:
            return []
        return [max(dets, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))]

    def detect_faces(self, frame, with_conf=False):
        results = self.detect(frame)
        if with_conf:
            return results
        return [(x1,y1,x2,y2) for (x1,y1,x2,y2,_) in results]

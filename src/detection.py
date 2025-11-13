from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# __file__ = .../src/detection.py
# parent        -> src
# parent.parent -> face-security-system  ✅ 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "yolov8_face.onnx"


class Detector:
    def __init__(self, model_path=None, conf_threshold: float = 0.4):
        """
        model_path:
          - 절대경로("/home/pi/.../yolov8_face.onnx")
          - 또는 "models/yolov8_face.onnx" 같은 상대경로
          - 또는 None이면 DEFAULT_MODEL_PATH 사용
        """
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not path.is_absolute():
            path = BASE_DIR / path

        self.model_path = str(path)
        self.conf_threshold = conf_threshold

        print(f"[Detector] Loading YOLO model via Ultralytics: {self.model_path}")

        # Ultralytics YOLO: ONNX 파일도 바로 로딩 가능
        self.model = YOLO(self.model_path)

    def detect(self, frame):
        """
        입력: BGR OpenCV 프레임 (numpy array)
        출력: [(x1, y1, x2, y2, conf), ...] 리스트
        """
        # Ultralytics가 알아서 전처리 + ONNXRuntime 추론 + 후처리까지 수행
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # boxes.xyxy: (N, 4), boxes.conf: (N,)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        detections = []
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            detections.append((int(x1), int(y1), int(x2), int(y2), float(c)))

        return detections

    def detect_faces(self, frame, with_conf: bool = False):
        """
        기존 코드 호환용 래퍼.
        - with_conf=False (기본): [(x1, y1, x2, y2), ...] 형태로 반환
        - with_conf=True : [(x1, y1, x2, y2, conf), ...] 그대로 반환
        """
        results = self.detect(frame)

        if with_conf:
            return results

        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _conf) in results]

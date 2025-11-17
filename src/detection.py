from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# __file__ = .../src/detection.py
# parent        -> src
# parent.parent -> face-security-system  ✅ 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "yolov8_face.onnx"
DEFAULT_HAILO_MODEL_PATH = BASE_DIR / "models" / "yolov8_face_hailo.hef"  # NEW


class Detector:
    def __init__(self, model_path=None, conf_threshold: float = 0.4,
                 backend: str = "cpu"):   # ✅ backend 추가
        """
        model_path:
          - 절대경로("/home/pi/.../yolov8_face.onnx")
          - 또는 "models/yolov8_face.onnx" 같은 상대경로
          - 또는 None이면 DEFAULT_MODEL_PATH 사용

        backend:
          - "cpu"   : Ultralytics + ONNXRuntime CPU
          - "hailo" : (나중에 Hailo YOLO 코드 붙일 자리)
        """
        self.backend = backend

        if backend == "cpu":
            path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
            if not path.is_absolute():
                path = BASE_DIR / path

            self.model_path = str(path)
            self.conf_threshold = conf_threshold

            print(f"[Detector] (CPU) Loading YOLO model via Ultralytics: {self.model_path}")
            self.model = YOLO(self.model_path)

        elif backend == "hailo":
            # TODO: 여기 나중에 Hailo YOLO 초기화 코드 넣기
            # ex) self.hailo_runner = MyHailoYoloRunner(DEFAULT_HAILO_MODEL_PATH)
            self.model_path = str(DEFAULT_HAILO_MODEL_PATH)
            self.conf_threshold = conf_threshold
            raise NotImplementedError("Hailo backend for Detector is not implemented yet.")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def detect(self, frame):
        """
        입력: BGR OpenCV 프레임 (numpy array)
        출력: [(x1, y1, x2, y2, conf), ...] 리스트
        """
        if self.backend == "cpu":
            results = self.model.predict(
                source=frame,
                imgsz=640,
                conf=self.conf_threshold,
                verbose=False,
            )[0]

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            detections = []
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                detections.append((int(x1), int(y1), int(x2), int(y2), float(c)))
            return detections

        elif self.backend == "hailo":
            # TODO: Hailo YOLO 추론 결과를 위와 같은 포맷으로 변환해서 리턴
            raise NotImplementedError("Hailo backend for Detector is not implemented yet.")

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

import numpy as np

class Detector:
    def __init__(self, backend: str = "cpu", model_path: str | None = None, hailo_model_path: str | None = None):
        """
        backend: "cpu" | "hailo"
        """
        self.backend = backend
        self.model_path = model_path
        self.hailo_model_path = hailo_model_path

        # TODO: 여기서 ONNXRuntime, HailoRT 등을 이용해 모델 로드
        # 지금은 더미 모드
        print(f"[Detector] init backend={backend}, model={model_path}, hailo={hailo_model_path}")

    def detect_faces(self, frame) -> list:
        """
        frame: BGR 이미지 (H x W x 3)
        return: [ (x1, y1, x2, y2), ... ]
        """
        h, w = frame.shape[:2]

        # TODO: 실제 YOLOv8 얼굴 검출 결과로 교체
        # 지금은 중앙에 더미 박스 하나 생성
        cx, cy = w // 2, h // 2
        box_w, box_h = w // 4, h // 4
        x1 = cx - box_w // 2
        y1 = cy - box_h // 2
        x2 = cx + box_w // 2
        y2 = cy + box_h // 2

        return [(x1, y1, x2, y2)]

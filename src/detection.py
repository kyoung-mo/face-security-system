from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ONNXRuntime (Hailo postprocess.onnx 실행용)
import onnxruntime as ort

# HailoRT Python API (hailo_platform) 시도 import
try:
    from hailo_platform import (
        HEF,
        Device,
        VDevice,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

# __file__ = .../src/detection.py
# parent        -> src
# parent.parent -> face-security-system  ✅ 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "yolov8_face.onnx"
DEFAULT_HAILO_MODEL_PATH = BASE_DIR / "models" / "yolov8_face_hailo.hef"  # Hailo용 HEF 경로
# postprocess ONNX (hailomz parse 시 생성된 파일)
DEFAULT_HAILO_POSTPROCESS_PATH = (
    BASE_DIR / "models" / "yolov8_face_parsed_dump" / "yolov8_face.postprocess.onnx"
)


class Detector:
    def __init__(self, model_path=None, conf_threshold: float = 0.2,
                 backend: str = "cpu"):
        """
        model_path:
          - 절대경로("/home/pi/.../yolov8_face.onnx")
          - 또는 "models/yolov8_face.onnx" 같은 상대경로
          - 또는 None이면 DEFAULT_MODEL_PATH 사용

        backend:
          - "cpu"   : Ultralytics + ONNXRuntime CPU
          - "hailo" : Hailo HEF 사용
        """
        self.backend = backend

        # -------------------
        # CPU (Ultralytics)
        # -------------------
        if backend == "cpu":
            path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
            if not path.is_absolute():
                path = BASE_DIR / path

            self.model_path = str(path)
            self.conf_threshold = conf_threshold

            print(f"[Detector] (CPU) Loading YOLO model via Ultralytics: {self.model_path}")
            self.model = YOLO(self.model_path)

        # -------------------
        # Hailo
        # -------------------
        elif backend == "hailo":
            if not HAILO_AVAILABLE:
                raise ImportError(
                    "hailo_platform Python 패키지를 찾을 수 없습니다. "
                    "HailoRT Python wheel을 설치한 뒤 다시 시도하세요."
                )

            # HEF 경로 설정
            path = Path(model_path) if model_path is not None else DEFAULT_HAILO_MODEL_PATH
            if not path.is_absolute():
                path = BASE_DIR / path
            self.model_path = str(path)
            self.conf_threshold = conf_threshold

            print(f"[Detector] (Hailo) Using HEF model: {self.model_path}")

            # 1) HEF 로드
            hef = HEF(self.model_path)

            # 2) Hailo 디바이스 스캔 및 VDevice 생성
            devices = Device.scan()
            if len(devices) == 0:
                raise RuntimeError("Hailo 장치를 찾을 수 없습니다. (Device.scan() 결과 0개)")

            self.vdevice = VDevice(device_ids=devices)

            # 3) 네트워크 그룹 설정 (PCIe 인터페이스)
            configure_params = ConfigureParams.create_from_hef(
                hef,
                interface=HailoStreamInterface.PCIe
            )
            self.network_group = self.vdevice.configure(hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            # 4) 입력 / 출력 vstream 정보 가져오기
            self.input_vstream_info = hef.get_input_vstream_infos()[0]
            self.output_vstream_info = hef.get_output_vstream_infos()[0]

            # ---- 입력 텐서 shape 분석 (HWC vs CHW) ----
            self.input_shape = tuple(self.input_vstream_info.shape)
            print(f"[Hailo] input_vstream shape = {self.input_shape}")

            if len(self.input_shape) != 3:
                raise RuntimeError(f"예상치 못한 입력 shape: {self.input_shape}")

            if 3 in self.input_shape:
                c_idx = self.input_shape.index(3)
                if c_idx == 0:
                    # (3, H, W) : CHW
                    self.hailo_format = "CHW"
                    self.input_height = self.input_shape[1]
                    self.input_width = self.input_shape[2]
                elif c_idx == 2:
                    # (H, W, 3) : HWC
                    self.hailo_format = "HWC"
                    self.input_height = self.input_shape[0]
                    self.input_width = self.input_shape[1]
                else:
                    raise RuntimeError(f"이상한 입력 shape: {self.input_shape}")
            else:
                self.hailo_format = "HWC"
                self.input_height = self.input_shape[0]
                self.input_width = self.input_shape[1]

            print(f"[Hailo] interpreted input as {self.hailo_format}, "
                  f"H={self.input_height}, W={self.input_width}")

            # 5) vstream 파라미터 생성 (float32, 비양자화 형태)
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=FormatType.FLOAT32,
            )
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=FormatType.FLOAT32,
            )

            # 6) Hailo에서 만든 postprocess ONNX 세션 준비
            post_path = DEFAULT_HAILO_POSTPROCESS_PATH
            if not post_path.exists():
                raise FileNotFoundError(f"Hailo postprocess ONNX를 찾을 수 없습니다: {post_path}")

            print(f"[Hailo] Using postprocess ONNX: {post_path}")
            self.post_sess = ort.InferenceSession(str(post_path), providers=["CPUExecutionProvider"])
            self.post_inp = self.post_sess.get_inputs()[0]
            self.post_inp_name = self.post_inp.name
            print(f"[Hailo] postprocess input name={self.post_inp_name}, shape={self.post_inp.shape}")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------
    # 공통 detect()
    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        입력: BGR OpenCV 프레임 (numpy array)
        출력: [(x1, y1, x2, y2, conf), ...] 리스트
        """

        # ============================================================
        # 1) CPU (Ultralytics YOLO)
        # ============================================================
        if self.backend == "cpu":
            results = self.model.predict(
                source=frame,
                imgsz=320,
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

            if len(detections) == 0:
                return []

            # 가장 큰 박스만 선택
            largest_det = max(
                detections,
                key=lambda d: (d[2] - d[0]) * (d[3] - d[1])  # w * h
            )

            return [largest_det]

        # ============================================================
        # 2) Hailo backend + postprocess ONNX
        # ============================================================
        elif self.backend == "hailo":

            # --- 1) Hailo 입력 전처리 ---
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            inp = resized.astype(np.float32)

            input_data = {
                self.input_vstream_info.name: np.expand_dims(inp, axis=0)
            }

            # --- 2) Hailo infer (raw head 출력) ---
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
                tf_nms_format=False,   # raw head
            ) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    output_data = infer_pipeline.infer(input_data)

            tensor = output_data[self.output_vstream_info.name]
            # 예: (1, 1, 2100, 65)
            raw = tensor
            while raw.ndim > 2:
                raw = raw.squeeze(0)  # -> (2100, 65)

            # Hailo postprocess ONNX 가 기대하는 입력 형태: (1, 65, 2100)
            raw_for_onnx = raw.T[np.newaxis, ...]  # (65,2100) -> (1,65,2100)

            # --- 3) postprocess ONNX 실행 (디코딩 + NMS 포함) ---
            post_out_list = self.post_sess.run(None, {self.post_inp_name: raw_for_onnx})
            post = post_out_list[0]

            # 보통 (1, N, 6) 이라고 가정 → squeeze 후 (N,6)
            while post.ndim > 2:
                post = post.squeeze(0)

            if post.size == 0:
                return []

            # 마지막 차원이 6 이상이라고 가정: [x1, y1, x2, y2, score, class]
            if post.shape[1] < 6:
                # 예상과 다르면 그냥 빈 결과 반환 (안전하게)
                print(f"[Hailo] Unexpected postprocess shape: {post.shape}")
                return []

            x1_in = post[:, 0]
            y1_in = post[:, 1]
            x2_in = post[:, 2]
            y2_in = post[:, 3]
            scores = post[:, 4]

            # Hailo postprocess 안에서 이미 conf threshold + NMS 했을 가능성이 큼.
            # 그래도 한 번 더 threshold 걸어줌.
            mask = scores >= self.conf_threshold
            if not np.any(mask):
                return []

            x1_in = x1_in[mask]
            y1_in = y1_in[mask]
            x2_in = x2_in[mask]
            y2_in = y2_in[mask]
            scores = scores[mask]

            # --- input(예: 320×320) → 원본 프레임 크기 변환 ---
            detections = []
            scale_x = w / float(self.input_width)
            scale_y = h / float(self.input_height)

            for x1i, y1i, x2i, y2i, score in zip(x1_in, y1_in, x2_in, y2_in, scores):
                x1 = int(x1i * scale_x)
                y1 = int(y1i * scale_y)
                x2 = int(x2i * scale_x)
                y2 = int(y2i * scale_y)

                # 범위 제한
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append((x1, y1, x2, y2, float(score)))

            if not detections:
                return []

            # 가장 큰 박스 1개만 사용 (CPU와 동일한 정책)
            largest_det = max(detections, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
            return [largest_det]

    # ------------------------------------------------------------------
    # 기존 코드 호환용 래퍼
    # ------------------------------------------------------------------
    def detect_faces(self, frame, with_conf: bool = False):
        """
        - with_conf=False (기본): [(x1, y1, x2, y2), ...]
        - with_conf=True : [(x1, y1, x2, y2, conf), ...]
        """
        results = self.detect(frame)

        if with_conf:
            return results

        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _conf) in results]

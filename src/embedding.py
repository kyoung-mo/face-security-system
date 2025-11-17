import os
import yaml
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

# config 경로 설정
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


class FaceEmbedder:
    def __init__(self, backend: str | None = None):
        """
        backend:
          - None이면 config.yaml의 runtime.backend 사용
          - "cpu"  : ONNX + ONNXRuntime CPU
          - "hailo": facenet_hailo.hef 사용 (나중에 구현)
        """
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        cfg_embed = config["models"]["embedding"]
        cfg_runtime = config.get("runtime", {})
        self.backend = backend or cfg_runtime.get("backend", "cpu")

        base_dir = Path(__file__).resolve().parent.parent  # 프로젝트 루트

        if self.backend == "cpu":
            # ONNX 모델 경로 설정
            model_path = base_dir / cfg_embed["onnx_path"]
            model_path = str(model_path)

            # ONNXRuntime 세션 로드
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # w600k_r50: 입력 112x112
            self.input_size = (112, 112)

            print(f"[FaceEmbedder] (CPU) ONNX model loaded: {model_path}")

        elif self.backend == "hailo":
            # TODO: 여기 Hailo 초기화 코드 넣기 (예: HailoRuntime, hef 로딩 등)
            hef_path = base_dir / cfg_embed["hailo_hef"]
            hef_path = str(hef_path)
            self.input_size = (112, 112)
            print(f"[FaceEmbedder] (Hailo) HEF model: {hef_path}")
            # self.hailo_runner = MyHailoFaceEmbedder(hef_path)
            raise NotImplementedError("Hailo backend for FaceEmbedder is not implemented yet.")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 얼굴 이미지(ROI)를 받아서 모델 입력 텐서(NCHW, float32)로 변환
        """
        img = cv2.resize(face_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지를 받아 L2-normalized 임베딩 벡터(512차원)를 반환
        """
        if self.backend == "cpu":
            input_tensor = self.preprocess(face_bgr)
            emb = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )[0]  # shape: (1, 512)
            emb = emb[0]  # (512,)

            # L2 정규화
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb

        elif self.backend == "hailo":
            # TODO: Hailo 러너에서 임베딩 얻기
            # emb = self.hailo_runner.get_embedding(face_bgr)
            # return emb
            raise NotImplementedError("Hailo backend for FaceEmbedder is not implemented yet.")

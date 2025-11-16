import os
import yaml
import numpy as np
import onnxruntime as ort
import cv2

# config 경로 설정
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    config["models"]["embedding"]["onnx_path"],
)

class FaceEmbedder:
    def __init__(self):
        # ONNXRuntime 세션 로드
        self.session = ort.InferenceSession(
            EMBEDDING_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # w600k_r50: 입력 112x112
        self.input_size = (112, 112)

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 얼굴 이미지(ROI)를 받아서 모델 입력 텐서(NCHW, float32)로 변환
        """
        # 1) 크기 맞추기
        img = cv2.resize(face_bgr, self.input_size)

        # 2) BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3) [0, 1]로 스케일링
        img = img.astype(np.float32) / 255.0

        # 4) [-1, 1] 정규화 (ArcFace 계열에서 일반적으로 사용)
        img = (img - 0.5) / 0.5

        # 5) HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # 6) 배치 차원 추가: (1, C, H, W)
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지를 받아 L2-normalized 임베딩 벡터(512차원)를 반환
        """
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

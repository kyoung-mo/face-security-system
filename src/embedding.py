import numpy as np

class Embedder:
    def __init__(self, backend: str = "cpu", model_path: str | None = None, hailo_model_path: str | None = None, embedding_dim: int = 128):
        self.backend = backend
        self.model_path = model_path
        self.hailo_model_path = hailo_model_path
        self.embedding_dim = embedding_dim

        # TODO: ONNXRuntime 또는 HailoRT 기반 FaceNet 로드
        print(f"[Embedder] init backend={backend}, model={model_path}, hailo={hailo_model_path}, dim={embedding_dim}")

    def get_embedding(self, face_img) -> np.ndarray:
        """
        face_img: 전처리된 얼굴 이미지 (H x W x 3, float32)
        return: (embedding_dim,) 벡터
        """
        # TODO: 실제 FaceNet 추론 코드로 교체
        # 지금은 더미로 고정된 pseudo embedding 반환 (이미지 기반 간단 feature)
        flat = face_img.mean(axis=(0, 1))  # (3,)
        rng = np.random.default_rng(int(flat.sum() * 1e6) % (2**32 - 1))
        emb = rng.normal(size=(self.embedding_dim,))
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype("float32")

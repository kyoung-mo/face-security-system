import os
import json
import yaml
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_DIM = config["models"]["recognition"]["embedding_dim"]
THRESHOLD = config["models"]["recognition"]["threshold"]

EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "embeddings.json",
)

class FaceRecognizer:
    def __init__(self):
        self.embeddings = {}  # name -> np.array
        self._load_embeddings()

    def _load_embeddings(self):
        if not os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = {}
            return

        with open(EMBEDDINGS_PATH, "r") as f:
            data = json.load(f)

        for name, emb_list in data.items():
            arr = np.array(emb_list, dtype=np.float32)
            # 혹시 차원이 안 맞으면 스킵
            if arr.shape[0] == EMBEDDING_DIM:
                self.embeddings[name] = arr

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def recognize(self, query_emb: np.ndarray):
        """
        query_emb: np.array (512,)
        return: (best_name, best_dist)
                인식 실패 시: (None, best_dist)
        """
        if not self.embeddings:
            return None, None

        best_name = None
        best_dist = 1e9

        for name, reg_emb in self.embeddings.items():
            dist = self._euclidean_distance(query_emb, reg_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist < THRESHOLD:
            return best_name, best_dist
        else:
            return None, best_dist

    def save_embedding(self, name: str, emb: np.ndarray):
        """
        새로운 사람 등록 시, 임베딩 저장
        """
        # 기존 데이터 로드
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data[name] = emb.tolist()

        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        with open(EMBEDDINGS_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 메모리 내 딕셔너리도 업데이트
        self.embeddings[name] = emb

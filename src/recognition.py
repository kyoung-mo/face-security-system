import os
import json
import yaml
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_DIM = config["models"]["recognition"]["embedding_dim"]
THRESHOLD = config["models"]["recognition"]["threshold"]


class FaceRecognizer:
    def __init__(self, backend: str = "cpu"):
        """
        backend:
          - "cpu"   → embeddings_cpu.json 사용
          - "hailo" → embeddings_hailo.json 사용
        """
        self.backend = backend

        if backend == "hailo":
            emb_file = "embeddings_hailo.json"
        else:
            emb_file = "embeddings_cpu.json"

        self.embeddings_path = os.path.join(
            PROJECT_ROOT, "data", emb_file
        )

        self.embeddings = {}
        self._load_embeddings()

        print(f"[FaceRecognizer] backend={self.backend}, using file={self.embeddings_path}")

    def _load_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            self.embeddings = {}
            return

        with open(self.embeddings_path, "r") as f:
            data = json.load(f)

        for name, emb_list in data.items():
            arr = np.array(emb_list, dtype=np.float32)
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
        # backend별 파일(self.embeddings_path)에 저장
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data[name] = emb.tolist()

        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 메모리 내 딕셔너리도 업데이트
        self.embeddings[name] = emb
        print(f"[FaceRecognizer] Saved embedding for '{name}' to {self.embeddings_path}")

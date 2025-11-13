import json
from pathlib import Path
import numpy as np

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)

class Recognizer:
    def __init__(self, embeddings_path: str | Path, threshold: float = 0.5):
        self.embeddings_path = Path(embeddings_path)
        self.threshold = threshold
        self.embeddings = {}  # user_id -> np.ndarray
        self._load_embeddings()

    def _load_embeddings(self):
        if not self.embeddings_path.exists():
            self.embeddings = {}
            return
        with self.embeddings_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.embeddings = {uid: np.array(vec, dtype="float32") for uid, vec in raw.items()}
        print(f"[Recognizer] loaded {len(self.embeddings)} users from {self.embeddings_path}")

    def _save_embeddings(self):
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {uid: emb.tolist() for uid, emb in self.embeddings.items()}
        with self.embeddings_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def add_user_embedding(self, user_id: str, embedding: np.ndarray):
        self.embeddings[user_id] = embedding.astype("float32")
        self._save_embeddings()

    def recognize(self, embedding: np.ndarray):
        """
        return: (best_user_id | None, best_distance)
        """
        if not self.embeddings:
            return None, None

        best_user = None
        best_score = -1.0

        for uid, emb in self.embeddings.items():
            score = _cosine_similarity(embedding, emb)
            if score > best_score:
                best_score = score
                best_user = uid

        # cosine similarity → distance 느낌으로 사용 (1 - sim)
        distance = 1.0 - best_score
        if distance <= self.threshold:
            return best_user, distance
        else:
            return None, distance

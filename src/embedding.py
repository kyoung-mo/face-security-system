import os
import yaml
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


class FaceEmbedder:
    def __init__(self, backend: str | None = None):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        cfg_embed = config["models"]["embedding"]
        cfg_runtime = config.get("runtime", {})
        self.backend = backend or cfg_runtime.get("backend", "cpu")

        base_dir = Path(__file__).resolve().parent.parent

        if self.backend == "cpu":
            model_path = base_dir / cfg_embed["onnx_path"]
            model_path = str(model_path)

            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # MobileFaceNet: 112×112
            self.input_size = (112, 112)

            print(f"[FaceEmbedder] (CPU) ONNX model loaded: {model_path}")

        elif self.backend == "hailo":
            hef_path = base_dir / cfg_embed["hailo_hef"]
            hef_path = str(hef_path)
            # MobileFaceNet: 112×112
            self.input_size = (112, 112)
            print(f"[FaceEmbedder] (Hailo) HEF model: {hef_path}")
            raise NotImplementedError("Hailo backend for FaceEmbedder is not implemented yet.")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        MobileFaceNet 전처리:
        (x - 127.5) / 127.5  → [-1, 1] 범위
        """
        img = cv2.resize(face_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5          # ← MobileFaceNet 전처리
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)    # (1, 3, 112, 112)
        return img

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        if self.backend == "cpu":
            input_tensor = self.preprocess(face_bgr)
            emb = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )[0]  # (1, 512)
            emb = emb[0]  # (512,)

            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb

        elif self.backend == "hailo":
            raise NotImplementedError("Hailo backend for FaceEmbedder is not implemented yet.")
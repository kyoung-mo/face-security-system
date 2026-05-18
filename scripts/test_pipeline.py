import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detection import Detector
from embedding import FaceEmbedder
from recognition import FaceRecognizer
from utils.preprocess import crop_and_resize

BASE_DIR = Path(__file__).resolve().parent.parent


def test_register(image_path: str, name: str):
    print(f"\n{'='*40}")
    print(f"[등록 테스트] 이름: {name}, 이미지: {image_path}")

    detector = Detector(
        model_path=str(BASE_DIR / "models" / "yolov8_face.onnx"),
        backend="cpu"
    )
    embedder = FaceEmbedder(backend="cpu")
    recognizer = FaceRecognizer(backend="cpu")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 이미지를 읽을 수 없음: {image_path}")
        return

    print(f"[INFO] 이미지 shape: {frame.shape}")

    bboxes = detector.detect_faces(frame)
    if not bboxes:
        print("[ERROR] 얼굴을 감지하지 못했습니다.")
        return

    print(f"[INFO] 얼굴 감지 성공: {bboxes[0]}")

    face_img = crop_and_resize(frame, bboxes[0])
    emb = embedder.get_embedding(face_img)
    print(f"[INFO] 임베딩 shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")

    recognizer.save_embedding(name, emb)
    print(f"[INFO] '{name}' 등록 완료")


def test_recognize(image_path: str):
    print(f"\n{'='*40}")
    print(f"[인식 테스트] 이미지: {image_path}")

    detector = Detector(
        model_path=str(BASE_DIR / "models" / "yolov8_face.onnx"),
        backend="cpu"
    )
    embedder = FaceEmbedder(backend="cpu")
    recognizer = FaceRecognizer(backend="cpu")

    if not recognizer.embeddings:
        print("[ERROR] 등록된 사람이 없습니다. 먼저 --mode register 실행하세요.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 이미지를 읽을 수 없음: {image_path}")
        return

    bboxes = detector.detect_faces(frame)
    if not bboxes:
        print("[ERROR] 얼굴을 감지하지 못했습니다.")
        return

    face_img = crop_and_resize(frame, bboxes[0])
    emb = embedder.get_embedding(face_img)
    user_id, distance = recognizer.recognize(emb)

    if user_id:
        print(f"[결과] ✅ 인식 성공: {user_id} (거리: {distance:.4f})")
    else:
        print(f"[결과] ❌ 미등록자 (거리: {distance:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["register", "recognize"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", default="test_user")
    args = parser.parse_args()

    if args.mode == "register":
        test_register(args.image, args.name)
    else:
        test_recognize(args.image)

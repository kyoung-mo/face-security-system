import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from detection import Detector
from embedding import FaceEmbedder
from recognition import FaceRecognizer
from utils.preprocess import crop_and_resize

BASE_DIR = Path(__file__).resolve().parent.parent

detector   = Detector(model_path=str(BASE_DIR/"models"/"yolov8_face.onnx"), backend="cpu")
embedder   = FaceEmbedder(backend="cpu")
recognizer = FaceRecognizer(backend="cpu")

name = input("등록할 이름: ").strip()
cap  = cv2.VideoCapture(0)

embeddings = []
total = 5
print(f"\n총 {total}장을 찍습니다. 각 장마다 Enter를 눌러 촬영하세요.")

for i in range(total):
    while True:
        input(f"\n[{i+1}/{total}] 얼굴 각도를 맞추고 Enter를 누르세요...")
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패, 다시 시도하세요.")
            continue
        bboxes = detector.detect_faces(frame)
        if not bboxes:
            print("얼굴을 감지하지 못했습니다. 다시 시도하세요.")
            continue
        face_img = crop_and_resize(frame, bboxes[0])
        emb = embedder.get_embedding(face_img)
        embeddings.append(emb)
        print(f"✅ [{i+1}/{total}] 촬영 완료")
        break

cap.release()

if embeddings:
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    recognizer.save_embedding(name, mean_emb)
    print(f"\n'{name}' 등록 완료 ({len(embeddings)}장)")
else:
    print("등록 실패")

import cv2
import numpy as np

def crop_and_resize(frame, bbox, size=(160, 160)):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, size)
    return face

def normalize_face(face):
    # [0,255] → [0,1] → 평균/표준편차 정규화 등 필요 시 조정
    face = face.astype("float32") / 255.0
    # 여기서 추가 전처리 가능
    return face

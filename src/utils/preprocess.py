import cv2
import numpy as np


def preprocess_for_yolov8_face(image, img_size: int = 640):
    """
    YOLOv8n-face ONNX 입력 형식에 맞게 전처리.
    반환:
        blob  : (1, 3, img_size, img_size) float32
        ratio : 원본 -> 리사이즈 스케일
        pad   : (pad_x, pad_y) (왼쪽, 위쪽 패딩)
    """
    h, w = image.shape[:2]

    # 1) 스케일 계산
    scale = min(img_size / w, img_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 2) 리사이즈
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3) 패딩 (letterbox)
    pad_w, pad_h = img_size - new_w, img_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # 4) BGR -> RGB, HWC -> CHW, [0,1]
    img = padded[:, :, ::-1]          # BGR -> RGB
    img = img.transpose(2, 0, 1)      # HWC -> CHW
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(img, axis=0)  # (1,3,H,W)

    ratio = scale
    pad = (left, top)

    return blob, ratio, pad


def crop_and_resize(image, bbox, size: int = 160):
    """
    YOLO로 얻은 bbox를 이용해 얼굴 영역을 잘라내고
    Facenet 입력 크기(size x size)로 리사이즈.
    bbox: (x1, y1, x2, y2)
    반환: (size, size, 3) BGR 이미지
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # 정수 + 이미지 범위 안으로 클램프
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        # 박스가 이상하면 그냥 중앙 크롭
        cx, cy = w // 2, h // 2
        half = min(w, h) // 4
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half

    face = image[y1:y2, x1:x2]  # BGR
    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)
    return face


def normalize_face(face_img):
    """
    얼굴 이미지를 신경망 입력용으로 정규화.
    - 입력: (H, W, 3) BGR uint8
    - 출력: (H, W, 3) float32, [0,1] 범위
    (나중에 embedding.py에서 추가로 transpose / batch 차원 붙일 수 있음)
    """
    face = face_img.astype(np.float32) / 255.0
    return face

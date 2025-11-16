from pathlib import Path
import cv2
import numpy as np

from camera import Camera
from detection import Detector
from embedding import FaceEmbedder        # ⬅️ 변경
from recognition import FaceRecognizer    # ⬅️ 변경
from utils.preprocess import crop_and_resize  # ⬅️ normalize_face 제거
from utils.config_loader import load_yaml


def run_register_mode():
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    # rec_cfg = config["recognition"]   # ⬅️ 이제 직접 쓰진 않지만, 필요하면 남겨둬도 됨

    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        backend=cam_cfg.get("backend", "picamera2"),
    )

    detector = Detector(
        model_path=paths["models"]["yolov8_face_onnx"],
        conf_threshold=config["detection"].get("conf_threshold", 0.4),
    )

    # 🔹 새 임베더 / 리코그나이저
    #  - 모델 경로, embedding_dim, threshold 등은
    #    embedding.py / recognition.py 내부에서 config.yaml을 통해 처리
    embedder = FaceEmbedder()
    recognizer = FaceRecognizer()

    user_id = input("등록할 사용자 ID를 입력하세요: ").strip()
    if not user_id:
        print("유효하지 않은 ID입니다.")
        return

    num_samples = 5
    embeddings = []

    print(f"{user_id} 등록을 위해 얼굴을 카메라에 맞추고 엔터를 누르세요.")
    input()

    for i in range(num_samples):
        frame = camera.get_frame()
        if frame is None:
            print("카메라 프레임을 가져오지 못했습니다.")
            continue

        bboxes = detector.detect_faces(frame)
        if not bboxes:
            print("얼굴을 찾지 못했습니다. 다시 시도합니다.")
            cv2.imshow("register", frame)
            cv2.waitKey(500)
            continue

        # 첫 번째 얼굴만 사용
        bbox = bboxes[0]
        face_img = crop_and_resize(frame, bbox)
        if face_img is None:
            print("얼굴 crop 실패")
            continue

        # 🔴 예전: face_norm = normalize_face(face_img)
        # 🔵 지금: FaceEmbedder가 resize + 정규화까지 내부 처리
        emb = embedder.get_embedding(face_img)
        embeddings.append(emb)

        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("register", frame)
        cv2.waitKey(500)

    camera.release()
    cv2.destroyAllWindows()

    if not embeddings:
        print("등록에 실패했습니다. 유효한 임베딩이 없습니다.")
        return

    # 샘플들의 평균 임베딩 + L2 정규화
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    # 🔹 새 FaceRecognizer의 저장 메서드 사용
    recognizer.save_embedding(user_id, mean_emb)
    print(f"{user_id} 등록 완료. ({len(embeddings)} 샘플 사용)")

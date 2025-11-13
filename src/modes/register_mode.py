from pathlib import Path
import cv2
import numpy as np

from camera import Camera
from detection import Detector
from embedding import Embedder
from recognition import Recognizer
from utils.preprocess import crop_and_resize, normalize_face
from utils.config_loader import load_yaml

def run_register_mode():
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    rec_cfg = config["recognition"]

    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
    )

    detector = Detector(
        backend="cpu",
        model_path=paths["models"]["yolov8_face_onnx"],
    )

    embedder = Embedder(
        backend="cpu",
        model_path=paths["models"]["facenet_onnx"],
        embedding_dim=rec_cfg.get("embedding_dim", 128),
    )

    recognizer = Recognizer(
        embeddings_path=paths["data"]["embeddings"],
        threshold=rec_cfg.get("distance_threshold", 0.5),
    )

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
        face_img = crop_and_resize(frame, bboxes[0])
        if face_img is None:
            print("얼굴 crop 실패")
            continue

        face_norm = normalize_face(face_img)
        emb = embedder.get_embedding(face_norm)
        embeddings.append(emb)

        cv2.rectangle(frame, (bboxes[0][0], bboxes[0][1]), (bboxes[0][2], bboxes[0][3]), (0,255,0), 2)
        cv2.imshow("register", frame)
        cv2.waitKey(500)

    camera.release()
    cv2.destroyAllWindows()

    if not embeddings:
        print("등록에 실패했습니다. 유효한 임베딩이 없습니다.")
        return

    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    recognizer.add_user_embedding(user_id, mean_emb)
    print(f"{user_id} 등록 완료. ({len(embeddings)} 샘플 사용)")

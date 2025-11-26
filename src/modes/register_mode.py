from pathlib import Path
import cv2
import numpy as np

from camera import Camera
from detection import Detector
from embedding import FaceEmbedder
from recognition import FaceRecognizer
from utils.preprocess import crop_and_resize
from utils.config_loader import load_yaml

# 프로젝트 루트 (face-security-system/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_register_mode(detector_backend: str = "cpu"):
    """
    detector_backend:
      - "cpu"   : yolov8_face.onnx + embeddings_cpu.json
      - "hailo" : yolov8_face_hailo.hef + embeddings_hailo.json
    """
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    runtime_cfg = config.get("runtime", {})

    # 만약 None이 들어오면 config.runtime.backend 사용 (안 들어오면 기본 cpu)
    if detector_backend is None:
        detector_backend = runtime_cfg.get("backend", "cpu")

    # ─────────────────────────────────────────────
    # data/registered_faces 절대 경로
    # ─────────────────────────────────────────────
    data_paths = paths.get("data", {})
    reg_rel = data_paths.get("registered_faces_dir", "data/registered_faces")
    reg_base_dir = PROJECT_ROOT / reg_rel
    reg_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] PROJECT_ROOT       = {PROJECT_ROOT}")
    print(f"[DEBUG] BASE REGISTER DIR  = {reg_base_dir}")
    print(f"[DEBUG] detector_backend   = {detector_backend}")

    # ─────────────────────────────────────────────
    # 카메라
    # ─────────────────────────────────────────────
    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        backend=cam_cfg.get("backend", "picamera2"),
    )

    # ─────────────────────────────────────────────
    # Detector: backend에 따라 모델 경로 선택
    #  - cpu   → paths.yaml에 있는 ONNX 경로 사용
    #  - hailo → detection.py의 DEFAULT_HAILO_MODEL_PATH 사용 (model_path=None)
    # ─────────────────────────────────────────────
    if detector_backend == "hailo":
        det_model_path = None  # detection.py 내부 기본값: models/yolov8_face_hailo.hef
    else:
        det_model_path = paths["models"]["yolov8_face_onnx"]

    detector = Detector(
        model_path=det_model_path,
        conf_threshold=config["detection"].get("conf_threshold", 0.4),
        backend=detector_backend,
    )

    # ─────────────────────────────────────────────
    # Embedder / Recognizer
    #  - FaceEmbedder는 아직 CPU ONNX만 사용
    #  - Recognizer는 backend에 따라
    #    embeddings_cpu.json / embeddings_hailo.json 분리 저장
    # ─────────────────────────────────────────────
    embedder_backend = "cpu"
    embedder = FaceEmbedder(backend="cpu")
    recognizer = FaceRecognizer(backend=detector_backend)

    # ─────────────────────────────────────────────
    # 사용자 입력 → 개인 폴더 생성
    # ─────────────────────────────────────────────
    user_id = input("등록할 사용자 ID를 입력하세요: ").strip()
    if not user_id:
        print("유효하지 않은 ID입니다.")
        camera.release()
        return

    user_dir = reg_base_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 5
    embeddings = []

    print(f"{user_id} 등록을 위해 얼굴을 카메라에 맞추고 엔터를 누르세요.")
    input()

    # ─────────────────────────────────────────────
    # 얼굴 캡처 반복 (5장)
    # ─────────────────────────────────────────────
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

        bbox = bboxes[0]
        face_img = crop_and_resize(frame, bbox)
        if face_img is None:
            print("얼굴 crop 실패")
            continue

        # 이미지 저장
        save_path = user_dir / f"{i + 1}.jpg"
        cv2.imwrite(str(save_path), face_img)
        print(f"[INFO] 저장: {save_path}")

        # 임베딩 추출
        emb = embedder.get_embedding(face_img)
        embeddings.append(emb)

        # 박스 표시
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("register", frame)
        cv2.waitKey(500)

    camera.release()
    cv2.destroyAllWindows()

    # ─────────────────────────────────────────────
    # 임베딩 평균 → backend 전용 embeddings_*.json에 저장
    # ─────────────────────────────────────────────
    if not embeddings:
        print("등록에 실패했습니다. 유효한 임베딩이 없습니다.")
        return

    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    recognizer.save_embedding(user_id, mean_emb)
    print(f"{user_id} 등록 완료. ({len(embeddings)} 샘플 사용)")
    print(f"저장된 이미지 경로: {user_dir}")

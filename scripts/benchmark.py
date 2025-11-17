#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import argparse

import numpy as np

# ─────────────────────────────────────────────
# 1. 프로젝트 루트 / src 경로 설정
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# src 모듈 import
from camera import Camera
from detection import Detector
from embedding import FaceEmbedder
from recognition import FaceRecognizer
from utils.preprocess import crop_and_resize
from utils.config_loader import load_yaml


def benchmark_pipeline(num_frames: int = 100, show_progress: bool = True):
    """
    카메라 → 얼굴 검출 → 얼굴 crop → 임베딩 → 인식
    전체 파이프라인의 FPS와 단계별 평균 시간을 측정한다.

    나중에 Hailo 환경에서 config/paths.yaml 및 backend만 바꾸고
    같은 스크립트를 실행해서 CPU vs Hailo 비교 가능.
    """

    # ─────────────────────────────────────────
    # 설정 로드
    # ─────────────────────────────────────────
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    det_cfg = config["detection"]

    # 카메라, 디텍터, 임베더, 리코그나이저 준비
    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        backend=cam_cfg.get("backend", "picamera2"),
    )

    detector = Detector(
        model_path=paths["models"]["yolov8_face_onnx"],
        conf_threshold=det_cfg.get("conf_threshold", 0.4),
        backend=backend,
    )

    embedder = FaceEmbedder(backend=backend)
    recognizer = FaceRecognizer()

    # ─────────────────────────────────────────
    # 벤치마크용 누적 변수
    # ─────────────────────────────────────────
    t_capture_list = []
    t_detect_list = []
    t_embed_recog_list = []

    total_start = time.perf_counter()
    processed_frames = 0
    face_found_frames = 0

    # ─────────────────────────────────────────
    # 프레임 루프
    # ─────────────────────────────────────────
    for i in range(num_frames):
        # 1) 캡처
        t0 = time.perf_counter()
        frame = camera.get_frame()
        t1 = time.perf_counter()

        if frame is None:
            continue

        t_capture_list.append(t1 - t0)
        processed_frames += 1

        # 2) 얼굴 검출
        t2 = time.perf_counter()
        bboxes = detector.detect_faces(frame)
        t3 = time.perf_counter()
        t_detect_list.append(t3 - t2)

        if not bboxes:
            if show_progress:
                print(f"[{i+1}/{num_frames}] No face detected.")
            continue

        face_found_frames += 1

        bbox = bboxes[0]
        face_img = crop_and_resize(frame, bbox)
        if face_img is None:
            if show_progress:
                print(f"[{i+1}/{num_frames}] Face crop failed.")
            continue

        # 3) 임베딩 + 인식 (한 덩어리로 측정)
        t4 = time.perf_counter()
        emb = embedder.get_embedding(face_img)
        user_id, distance = recognizer.recognize(emb)
        t5 = time.perf_counter()
        t_embed_recog_list.append(t5 - t4)

        if show_progress:
            print(
                f"[{i+1}/{num_frames}] "
                f"face_found={user_id is not None}, user_id={user_id}, distance={distance}"
            )

    total_end = time.perf_counter()
    total_elapsed = total_end - total_start

    camera.release()

    # ─────────────────────────────────────────
    # 결과 집계
    # ─────────────────────────────────────────
    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    avg_capture = avg(t_capture_list)
    avg_detect = avg(t_detect_list)
    avg_embed_recog = avg(t_embed_recog_list)

    fps_overall = processed_frames / total_elapsed if total_elapsed > 0 else 0.0

    print("\n========== BENCHMARK RESULT ==========")
    print(f"전체 수행 시간         : {total_elapsed:.3f} 초")
    print(f"처리된 프레임 수       : {processed_frames} / 요청 {num_frames}")
    print(f"얼굴이 검출된 프레임 수: {face_found_frames}")
    print(f"전체 파이프라인 FPS    : {fps_overall:.2f} FPS\n")

    print("단계별 평균 시간 (1 프레임 기준):")
    print(f"  - 캡처               : {avg_capture*1000:.2f} ms")
    print(f"  - 얼굴 검출          : {avg_detect*1000:.2f} ms")
    print(f"  - 임베딩+인식 단계   : {avg_embed_recog*1000:.2f} ms (얼굴 검출된 프레임 기준)\n")

    print("주의: 임베딩+인식 시간은 얼굴이 검출된 프레임들에 대해서만 평균 냄.")
    print("      나중에 Hailo 환경에서 같은 스크립트 실행해서 수치 비교하면 됨.")
    print("=======================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Face-security-system 성능 벤치마크"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cpu",
        choices=["cpu", "hailo"],
        help="backend 선택 (cpu | hailo)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="벤치마크 프레임 수",
    )
    args = parser.parse_args()

    print(f"[INFO] backend mode  : {args.mode}")
    print(f"[INFO] frames       : {args.frames}")

    benchmark_pipeline(num_frames=args.frames, backend=args.mode)

if __name__ == "__main__":
    main()

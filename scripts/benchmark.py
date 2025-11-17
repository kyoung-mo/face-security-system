#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import argparse
import csv
from datetime import datetime
import subprocess
import re

import numpy as np

# psutil은 있으면 사용, 없으면 CPU 사용률은 -1로 기록
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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


def read_cpu_temperature() -> float | None:
    """
    라즈베리파이 CPU 온도(섭씨)를 읽어서 반환.
    읽기 실패 시 None 반환.
    """
    # 1) /sys/class/thermal 경로 시도
    candidates = [
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/class/thermal/thermal_zone1/temp"),
    ]
    for p in candidates:
        if p.exists():
            try:
                v = p.read_text().strip()
                mv = float(v)
                # 보통 millidegree 단위로 들어오니 1000으로 나눔
                return mv / 1000.0 if mv > 200 else mv
            except Exception:
                continue

    # 2) vcgencmd measure_temp 시도
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r"([\d\.]+)", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass

    return None


def benchmark_pipeline(num_frames: int = 100, backend: str = "cpu", show_progress: bool = True):
    """
    카메라 → 얼굴 검출 → 얼굴 crop → 임베딩 → 인식
    전체 파이프라인의 FPS와 단계별 평균 시간을 측정한다.

    같은 스크립트를 CPU / Hailo 환경에서 실행하여
    성능을 비교할 수 있도록 설계.
    """

    # ─────────────────────────────────────────
    # 설정 로드
    # ─────────────────────────────────────────
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    det_cfg = config["detection"]

    cam_width = cam_cfg.get("width", 640)
    cam_height = cam_cfg.get("height", 480)

    # 카메라, 디텍터, 임베더, 리코그나이저 준비
    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_width,
        height=cam_height,
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

    cpu_usage_list = []   # 시스템 전체 CPU 사용률 (%)
    cpu_temp_list = []    # CPU 온도 (°C)

    if PSUTIL_AVAILABLE:
        # 첫 호출은 기준값 맞추기용 (버려도 됨)
        psutil.cpu_percent(interval=None)

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
            if show_progress:
                print(f"[{i+1}/{num_frames}] frame is None (capture failed).")
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
        else:
            face_found_frames += 1

            bbox = bboxes[0]
            face_img = crop_and_resize(frame, bbox)
            if face_img is None:
                if show_progress:
                    print(f"[{i+1}/{num_frames}] Face crop failed.")
            else:
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

        # ─────────────────────────────────────
        # CPU 사용률 / 온도 샘플링
        # ─────────────────────────────────────
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=None)
            cpu_usage_list.append(cpu_usage)

        temp = read_cpu_temperature()
        if temp is not None:
            cpu_temp_list.append(temp)

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
    face_found_ratio = (
        face_found_frames / processed_frames if processed_frames > 0 else 0.0
    )

    avg_cpu_usage = avg(cpu_usage_list) if cpu_usage_list else -1.0
    avg_cpu_temp = avg(cpu_temp_list) if cpu_temp_list else -1.0
    max_cpu_temp = max(cpu_temp_list) if cpu_temp_list else -1.0

    print("\n========== BENCHMARK RESULT ==========")
    print(f"backend 모드            : {backend}")
    print(f"전체 수행 시간         : {total_elapsed:.3f} 초")
    print(f"요청 프레임 수         : {num_frames}")
    print(f"처리된 프레임 수       : {processed_frames}")
    print(f"얼굴이 검출된 프레임 수: {face_found_frames} "
          f"({face_found_ratio*100:.1f} %)")
    print(f"전체 파이프라인 FPS    : {fps_overall:.2f} FPS\n")

    print("단계별 평균 시간 (1 프레임 기준):")
    print(f"  - 캡처               : {avg_capture*1000:.2f} ms")
    print(f"  - 얼굴 검출          : {avg_detect*1000:.2f} ms")
    print(f"  - 임베딩+인식 단계   : {avg_embed_recog*1000:.2f} ms "
          f"(얼굴 검출된 프레임 기준)\n")

    print("시스템 자원 (실행 중 샘플 평균):")
    if avg_cpu_usage >= 0:
        print(f"  - 평균 CPU 사용률    : {avg_cpu_usage:.1f} %")
    else:
        print("  - 평균 CPU 사용률    : (psutil 미설치로 측정 불가)")
    if avg_cpu_temp >= 0:
        print(f"  - 평균 CPU 온도      : {avg_cpu_temp:.1f} °C")
        print(f"  - 최대 CPU 온도      : {max_cpu_temp:.1f} °C")
    else:
        print("  - CPU 온도           : (측정 불가)")

    print("\n주의: 임베딩+인식 시간은 얼굴이 검출된 프레임들에 대해서만 평균 냄.")
    print("      나중에 Hailo 환경에서 같은 스크립트 실행해서 수치 비교하면 됨.")
    print("=======================================\n")

    # ─────────────────────────────────────────
    # CSV로 결과 저장
    # ─────────────────────────────────────────
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / f"benchmark_{backend}.csv"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "num_frames_requested": num_frames,
        "processed_frames": processed_frames,
        "face_found_frames": face_found_frames,
        "face_found_ratio": face_found_ratio,
        "total_elapsed_sec": total_elapsed,
        "overall_fps": fps_overall,
        "avg_capture_ms": avg_capture * 1000.0,
        "avg_detect_ms": avg_detect * 1000.0,
        "avg_embed_recog_ms": avg_embed_recog * 1000.0,
        "camera_width": cam_width,
        "camera_height": cam_height,
        "avg_cpu_usage_percent": avg_cpu_usage,
        "avg_cpu_temp_c": avg_cpu_temp,
        "max_cpu_temp_c": max_cpu_temp,
    }

    fieldnames = [
        "timestamp",
        "backend",
        "num_frames_requested",
        "processed_frames",
        "face_found_frames",
        "face_found_ratio",
        "total_elapsed_sec",
        "overall_fps",
        "avg_capture_ms",
        "avg_detect_ms",
        "avg_embed_recog_ms",
        "camera_width",
        "camera_height",
        "avg_cpu_usage_percent",
        "avg_cpu_temp_c",
        "max_cpu_temp_c",
    ]

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[INFO] 벤치마크 결과를 '{csv_path}' 에 저장했습니다.")


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

    print(f"[INFO] backend mode : {args.mode}")
    print(f"[INFO] frames       : {args.frames}")

    benchmark_pipeline(num_frames=args.frames, backend=args.mode)


if __name__ == "__main__":
    main()

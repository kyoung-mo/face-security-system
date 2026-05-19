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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from camera import Camera
from recognition import FaceRecognizer
from utils.preprocess import crop_and_resize
from utils.config_loader import load_yaml


def read_cpu_temperature() -> float | None:
    candidates = [
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/class/thermal/thermal_zone1/temp"),
    ]
    for p in candidates:
        if p.exists():
            try:
                v = p.read_text().strip()
                mv = float(v)
                return mv / 1000.0 if mv > 200 else mv
            except Exception:
                continue
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

    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    det_cfg = config["detection"]
    cam_width = cam_cfg.get("width", 640)
    cam_height = cam_cfg.get("height", 480)

    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_width,
        height=cam_height,
        backend=cam_cfg.get("backend", "picamera2"),
    )

    if backend == "hailo":
        from detection_hef import Detector
        from embedding_hef import FaceEmbedderHEF as FaceEmbedder
        detector = Detector(
            model_path=str(PROJECT_ROOT / "models" / "yolov8_face_zoo.hef"),
            conf_threshold=det_cfg.get("conf_threshold", 0.4),
        )
        embedder = FaceEmbedder(
            model_path=str(PROJECT_ROOT / "models" / "mobilefacenet_zoo.hef"),
        )
    else:
        from detection import Detector
        from embedding import FaceEmbedder
        detector = Detector(
            model_path=str(PROJECT_ROOT / "models" / "yolov8_face.onnx"),
            conf_threshold=det_cfg.get("conf_threshold", 0.4),
        )
        embedder = FaceEmbedder(backend="cpu")
    recognizer = FaceRecognizer(backend=backend)

    # ── 누적 변수 ──────────────────────────────────
    t_capture_list    = []
    t_detect_list     = []
    t_embed_list      = []   # ★ 임베딩만
    t_recog_list      = []   # ★ 인식(거리계산)만
    cpu_usage_list    = []
    cpu_temp_list     = []
    mem_usage_list    = []   # ★ 메모리(MB)
    distances         = []   # ★ 인식 거리 분포

    if PSUTIL_AVAILABLE:
        psutil.cpu_percent(interval=None)
        proc = psutil.Process()

    total_start = time.perf_counter()
    processed_frames  = 0
    face_found_frames = 0

    for i in range(num_frames):
        # 1) 캡처
        t0 = time.perf_counter()
        frame = camera.get_frame()
        t1 = time.perf_counter()

        if frame is None:
            if show_progress:
                print(f"[{i+1}/{num_frames}] frame is None.")
            continue

        t_capture_list.append(t1 - t0)
        processed_frames += 1

        # 2) 얼굴 감지
        t2 = time.perf_counter()
        bboxes = detector.detect_faces(frame)
        t3 = time.perf_counter()
        t_detect_list.append(t3 - t2)

        if not bboxes:
            if show_progress:
                print(f"[{i+1}/{num_frames}] No face detected.")
        else:
            face_found_frames += 1
            bbox     = bboxes[0]
            face_img = crop_and_resize(frame, bbox)

            if face_img is not None:
                # 3) 임베딩만 측정 ★
                t4 = time.perf_counter()
                emb = embedder.get_embedding(face_img)
                t5 = time.perf_counter()
                t_embed_list.append(t5 - t4)

                # 4) 인식(거리계산)만 측정 ★
                t6 = time.perf_counter()
                user_id, distance = recognizer.recognize(emb)
                t7 = time.perf_counter()
                t_recog_list.append(t7 - t6)

                if distance is not None:
                    distances.append(distance)

                if show_progress:
                    print(
                        f"[{i+1}/{num_frames}] "
                        f"detect={t3-t2:.3f}s  embed={t5-t4:.3f}s  recog={t7-t6:.3f}s  "
                        f"user={user_id}  dist={distance:.4f}"
                    )

        # 시스템 자원 샘플링
        if PSUTIL_AVAILABLE:
            cpu_usage_list.append(psutil.cpu_percent(interval=None))
            mem_usage_list.append(proc.memory_info().rss / 1024 / 1024)  # MB

        temp = read_cpu_temperature()
        if temp is not None:
            cpu_temp_list.append(temp)

    total_end     = time.perf_counter()
    total_elapsed = total_end - total_start
    camera.release()

    # ── 통계 함수 ──────────────────────────────────
    def stats(lst):
        if not lst:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        a = np.array(lst)
        return (
            float(np.mean(a)),
            float(np.min(a)),
            float(np.max(a)),
            float(np.std(a)),
            float(np.percentile(a, 95)),   # P95
        )

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    fps_overall      = processed_frames / total_elapsed if total_elapsed > 0 else 0.0
    face_found_ratio = face_found_frames / processed_frames if processed_frames > 0 else 0.0

    det_mean, det_min, det_max, det_std, det_p95       = stats(t_detect_list)
    emb_mean, emb_min, emb_max, emb_std, emb_p95       = stats(t_embed_list)
    rec_mean, rec_min, rec_max, rec_std, rec_p95       = stats(t_recog_list)
    dist_mean, dist_min, dist_max, dist_std, dist_p95  = stats(distances)

    avg_cpu   = avg(cpu_usage_list) if cpu_usage_list else -1.0
    avg_temp  = avg(cpu_temp_list)  if cpu_temp_list  else -1.0
    max_temp  = max(cpu_temp_list)  if cpu_temp_list  else -1.0
    avg_mem   = avg(mem_usage_list) if mem_usage_list else -1.0
    max_mem   = max(mem_usage_list) if mem_usage_list else -1.0

    # ── 출력 ───────────────────────────────────────
    print("\n========== BENCHMARK RESULT ==========")
    print(f"backend                : {backend}")
    print(f"전체 수행 시간         : {total_elapsed:.3f} 초")
    print(f"요청/처리 프레임       : {num_frames} / {processed_frames}")
    print(f"얼굴 검출 프레임       : {face_found_frames} ({face_found_ratio*100:.1f}%)")
    print(f"전체 FPS               : {fps_overall:.2f} FPS")

    print("\n─── 단계별 추론 시간 (ms) ───────────────")
    print(f"{'항목':<14} {'평균':>8} {'최소':>8} {'최대':>8} {'표준편차':>10} {'P95':>8}")
    print(f"{'얼굴 감지':<14} {det_mean*1000:>8.2f} {det_min*1000:>8.2f} {det_max*1000:>8.2f} {det_std*1000:>10.2f} {det_p95*1000:>8.2f}")
    print(f"{'임베딩':<14} {emb_mean*1000:>8.2f} {emb_min*1000:>8.2f} {emb_max*1000:>8.2f} {emb_std*1000:>10.2f} {emb_p95*1000:>8.2f}")
    print(f"{'인식(거리)':<14} {rec_mean*1000:>8.2f} {rec_min*1000:>8.2f} {rec_max*1000:>8.2f} {rec_std*1000:>10.2f} {rec_p95*1000:>8.2f}")

    print("\n─── 인식 거리 분포 ──────────────────────")
    print(f"  평균: {dist_mean:.4f}  최소: {dist_min:.4f}  최대: {dist_max:.4f}  표준편차: {dist_std:.4f}  P95: {dist_p95:.4f}")
    print(f"  (threshold={load_yaml('config/config.yaml')['models']['recognition']['threshold']})")

    print("\n─── 시스템 자원 ─────────────────────────")
    if avg_cpu >= 0:
        print(f"  평균 CPU 사용률    : {avg_cpu:.1f}%")
    if avg_temp >= 0:
        print(f"  평균/최대 CPU 온도 : {avg_temp:.1f}°C / {max_temp:.1f}°C")
    if avg_mem >= 0:
        print(f"  평균/최대 메모리   : {avg_mem:.1f}MB / {max_mem:.1f}MB")
    print("=======================================\n")

    # ── CSV 저장 ───────────────────────────────────
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"benchmark_{backend}.csv"

    row = {
        "timestamp"              : datetime.now().isoformat(timespec="seconds"),
        "backend"                : backend,
        "num_frames_requested"   : num_frames,
        "processed_frames"       : processed_frames,
        "face_found_frames"      : face_found_frames,
        "face_found_ratio"       : face_found_ratio,
        "total_elapsed_sec"      : total_elapsed,
        "overall_fps"            : fps_overall,
        # 얼굴 감지
        "detect_mean_ms"         : det_mean * 1000,
        "detect_min_ms"          : det_min  * 1000,
        "detect_max_ms"          : det_max  * 1000,
        "detect_std_ms"          : det_std  * 1000,
        "detect_p95_ms"          : det_p95  * 1000,
        # 임베딩
        "embed_mean_ms"          : emb_mean * 1000,
        "embed_min_ms"           : emb_min  * 1000,
        "embed_max_ms"           : emb_max  * 1000,
        "embed_std_ms"           : emb_std  * 1000,
        "embed_p95_ms"           : emb_p95  * 1000,
        # 인식
        "recog_mean_ms"          : rec_mean * 1000,
        "recog_min_ms"           : rec_min  * 1000,
        "recog_max_ms"           : rec_max  * 1000,
        "recog_std_ms"           : rec_std  * 1000,
        "recog_p95_ms"           : rec_p95  * 1000,
        # 인식 거리
        "dist_mean"              : dist_mean,
        "dist_min"               : dist_min,
        "dist_max"               : dist_max,
        "dist_std"               : dist_std,
        "dist_p95"               : dist_p95,
        # 시스템
        "avg_cpu_usage_percent"  : avg_cpu,
        "avg_cpu_temp_c"         : avg_temp,
        "max_cpu_temp_c"         : max_temp,
        "avg_mem_mb"             : avg_mem,
        "max_mem_mb"             : max_mem,
        "camera_width"           : cam_width,
        "camera_height"          : cam_height,
    }

    fieldnames = list(row.keys())
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[INFO] 결과 저장: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Face-security-system 성능 벤치마크")
    parser.add_argument("--mode",   type=str, default="cpu", choices=["cpu", "hailo"])
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    print(f"[INFO] backend : {args.mode}")
    print(f"[INFO] frames  : {args.frames}")
    benchmark_pipeline(num_frames=args.frames, backend=args.mode)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import cv2

from camera import Camera
from detection import Detector
from utils.config_loader import load_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Simple detection test (CPU / Hailo)"
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "hailo"],
        default="hailo",
        help="사용할 백엔드 선택 (cpu | hailo)",
    )
    args = parser.parse_args()
    backend = args.backend

    # ───────────────────────────────────
    # 설정 로드
    # ───────────────────────────────────
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    det_cfg = config["detection"]

    # ───────────────────────────────────
    # 카메라 생성
    # ───────────────────────────────────
    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        backend=cam_cfg.get("backend", "picamera2"),
        detector_backend=backend,
    )

    # ───────────────────────────────────
    # Detector 생성 (CPU / Hailo)
    # ───────────────────────────────────
    if backend == "hailo":
        # paths.yaml에 yolov8_face_hailo_hef가 있으면 사용, 없으면 None → detection.py 기본값 사용
        det_model_path = paths["models"].get("yolov8_face_hailo_hef", None)
    else:
        det_model_path = paths["models"]["yolov8_face_onnx"]

    detector = Detector(
        model_path=det_model_path,
        conf_threshold=det_cfg.get("conf_threshold", 0.4),
        backend=backend,
    )

    print(f"[detect_test] backend = {backend}")
    print("q 키를 누르면 종료합니다.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("프레임을 가져오지 못했습니다.")
                break

            # detection.py 의 detect()는 [(x1,y1,x2,y2,conf), ...] 반환
            # detection.py 의 detect()는 [(x1,y1,x2,y2,conf), ...] 반환
            detections = detector.detect(frame)

            h_img, w_img = frame.shape[:2]

            # 1) 기본 정보 찍기
            print("---- raw detections ----")
            for i, (x1, y1, x2, y2, conf) in enumerate(detections[:5]):
                w = x2 - x1
                h = y2 - y1
                ratio = w / (h + 1e-6)
                print(
                    f"{i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                    f"w={w}, h={h}, ratio={ratio:.2f}, conf={conf:.3f}"
                )

            # 2) 얼굴 후보 필터링
            face_candidates = []
            for (x1, y1, x2, y2, conf) in detections:
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                ratio = w / h

                # 너무 가로로 넓은 박스(예: 문, 벽 등)는 버리기
                if ratio > 1.5:
                    continue

                # 너무 작은 박스도 무시 (이미지 높이의 15% 미만이면 버림)
                if h < h_img * 0.15:
                    continue

                face_candidates.append((x1, y1, x2, y2, conf))

            # 후보가 하나도 없으면 그냥 원래 detections 전체 사용
            if face_candidates:
                detections_used = face_candidates
            else:
                detections_used = detections

            # 3) conf 기준으로 정렬
            detections_sorted = sorted(
                detections_used, key=lambda d: d[4], reverse=True
            )

            # 4) 프레임 중앙에 가장 가까운 박스를 "대표 얼굴"로 선택
            if detections_sorted:
                cx_img, cy_img = w_img / 2, h_img / 2

                def center_dist(box):
                    x1, y1, x2, y2, conf = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    return ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5

                best_box = min(detections_sorted, key=center_dist)
            else:
                best_box = None

            # 5) 화면에 박스 그리기
            for (x1, y1, x2, y2, conf) in detections_sorted:
                if best_box is not None and (x1, y1, x2, y2, conf) == best_box:
                    # 대표 얼굴 → 초록 두껍게
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    # 나머지 후보 → 노랑 얇게
                    color = (0, 255, 255)
                    thickness = 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("detect_test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

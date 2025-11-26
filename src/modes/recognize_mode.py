import cv2

from camera import Camera
from detection import Detector
from embedding import FaceEmbedder          # ✅ 새 클래스
from recognition import FaceRecognizer      # ✅ 새 클래스
from gpio_control import GPIOController, GPIOConfig
from lcd_display import LCDDisplay
from utils.preprocess import crop_and_resize  # ✅ normalize_face는 제거
from utils.config_loader import load_yaml
from utils.logging_utils import append_access_log


def run_recognize_mode(detector_backend="cpu"):
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    gpio_cfg = config["gpio"]
    log_cfg = config["logging"]
    det_cfg = config["detection"]
    
    print(f"[RecognizeMode] Detector backend = {detector_backend}")

    # 🔹 카메라 설정 그대로 사용 + Detector backend 추가
    camera = Camera(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        backend=cam_cfg.get("backend", "picamera2"),
        detector_backend = detector_backend    # 추가
    )

    # 🔹 얼굴 검출기: backend에 따라 onnx / hef 선택
    if detector_backend == "hailo":
        # detection.py 안에서 DEFAULT_HAILO_MODEL_PATH(hef) 사용
        det_model_path = None
    else:
        # CPU일 때는 ONNX 경로 사용
        det_model_path = paths["models"]["yolov8_face_onnx"]

    detector = Detector(
        model_path=det_model_path,
        conf_threshold=config["detection"].get("conf_threshold", 0.4),
        backend=detector_backend,
    )

    # 🔹 새 FaceEmbedder / FaceRecognizer
    #     - FaceEmbedder는 항상 CPU 사용 (Hailo 분기 아직 미구현)
    embedder = FaceEmbedder(backend="cpu")
    recognizer = FaceRecognizer(backend=detector_backend)

    # 🔹 GPIO / LCD 는 기존 그대로
    gpio = GPIOController(GPIOConfig(
        enabled=gpio_cfg.get("enabled", False),
        green_led_pin=gpio_cfg.get("green_led_pin", 17),
        red_led_pin=gpio_cfg.get("red_led_pin", 27),
        buzzer_pin=gpio_cfg.get("buzzer_pin", 22),
    ))

    lcd = LCDDisplay()

    print("실시간 인식 모드를 시작합니다. q를 눌러 종료하세요.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("프레임을 가져오지 못했습니다.")
                break

            bboxes = detector.detect_faces(frame)

            if not bboxes:
                lcd.show_text("No face detected")
                gpio.reset()
            else:
                # ✅ 첫 번째 얼굴만 처리 (기존과 동일)
                bbox = bboxes[0]  # [x1, y1, x2, y2, ...] 형태라고 가정
                face_img = crop_and_resize(frame, bbox)  # BGR 얼굴 ROI 리턴
                if face_img is None:
                    cv2.imshow("recognize", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # 🔴 예전: face_norm = normalize_face(face_img)
                # 🔵 지금: FaceEmbedder가 내부에서 resize + 정규화까지 수행함
                emb = embedder.get_embedding(face_img)

                user_id, distance = recognizer.recognize(emb)

                if user_id is not None:
                    line1 = "Access Granted"
                    line2 = f"{user_id} (d={distance:.3f})"
                    lcd.show_text(f"{line1}\n{line2}")
                    gpio.green_on()
                    gpio.red_off()
                    gpio.buzzer_off()
                    append_access_log(
                        log_cfg["access_log_path"],
                        user_id,
                        "granted",
                        distance,
                    )
                    color = (0, 255, 0)
                else:
                    line1 = "Access Denied"
                    if distance is not None:
                        # 등록 안 된 사람이라 user_id는 없으니 거리만
                        line2 = f"(d={distance:.3f})"
                    else:
                        line2 = ""
                    lcd.show_text(f"{line1}\n{line2}" if line2 else line1)
                    gpio.green_off()
                    gpio.red_on()
                    gpio.buzzer_on()
                    append_access_log(
                        log_cfg["access_log_path"],
                        None,
                        "denied",
                        distance,
                    )
                    color = (0, 0, 255)

                # 얼굴 박스 그림 (bbox 형태에 따라 인덱스 조정 필요)
                x1, y1, x2, y2 = bbox[:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("recognize", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        camera.release()
        gpio.reset()
        gpio.cleanup()
        cv2.destroyAllWindows()

        # 🔹 종료 시 LCD 초기화 (또는 중립 메시지)
        try:
            lcd.show_text("System Stopped")
            # 필요하면 잠깐 보여주고 지우고 싶으면:
            # import time; time.sleep(1)
            lcd.clear()
        except Exception as e:
            print(f"[LCD] cleanup error: {e}")

        cv2.destroyAllWindows()

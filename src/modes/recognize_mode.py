import cv2

from camera import Camera
from detection import Detector
from embedding import Embedder
from recognition import Recognizer
from gpio_control import GPIOController, GPIOConfig
from lcd_display import LCDDisplay
from utils.preprocess import crop_and_resize, normalize_face
from utils.config_loader import load_yaml
from utils.logging_utils import append_access_log

def run_recognize_mode():
    config = load_yaml("config/config.yaml")
    paths = load_yaml("config/paths.yaml")

    cam_cfg = config["camera"]
    rec_cfg = config["recognition"]
    gpio_cfg = config["gpio"]
    log_cfg = config["logging"]

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

    embedder = Embedder(
        backend="cpu",
        model_path=paths["models"]["facenet_onnx"],
        embedding_dim=rec_cfg.get("embedding_dim", 128),
    )

    recognizer = Recognizer(
        embeddings_path=paths["data"]["embeddings"],
        threshold=rec_cfg.get("distance_threshold", 0.5),
    )

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
                # 첫 번째 얼굴만 처리
                bbox = bboxes[0]
                face_img = crop_and_resize(frame, bbox)
                if face_img is None:
                    continue

                face_norm = normalize_face(face_img)
                emb = embedder.get_embedding(face_norm)
                user_id, distance = recognizer.recognize(emb)

                if user_id is not None:
                    msg = f"Access Granted: {user_id} (d={distance:.3f})"
                    lcd.show_text(msg)
                    gpio.green_on()
                    gpio.red_off()
                    gpio.buzzer_off()
                    append_access_log(log_cfg["access_log_path"], user_id, "granted", distance)
                    color = (0, 255, 0)
                else:
                    msg = f"Access Denied (d={distance:.3f})" if distance is not None else "Access Denied"
                    lcd.show_text(msg)
                    gpio.green_off()
                    gpio.red_on()
                    gpio.buzzer_on()
                    append_access_log(log_cfg["access_log_path"], None, "denied", distance)
                    color = (0, 0, 255)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            cv2.imshow("recognize", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        camera.release()
        gpio.reset()
        gpio.cleanup()
        cv2.destroyAllWindows()

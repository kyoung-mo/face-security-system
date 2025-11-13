import cv2

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class Camera:
    """
    라즈베리파이에서는 기본적으로 Picamera2를 사용하고,
    없거나 원하면 OpenCV VideoCapture로 fallback 하는 카메라 래퍼.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        backend: str = "picamera2",  # "picamera2" 또는 "opencv"
    ):
        self.width = width
        self.height = height
        self.backend = backend
        self.use_picamera2 = False
        self.cap = None
        self.picam2 = None

        # Picamera2 우선 사용
        if backend == "picamera2" and PICAMERA2_AVAILABLE:
            self._init_picamera2()
        else:
            self._init_opencv(device_index)

    # ----------------- 내부 초기화 함수들 ----------------- #

    def _init_picamera2(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.use_picamera2 = True
        print("[Camera] Picamera2 backend 사용 중")

    def _init_opencv(self, device_index: int):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.use_picamera2 = False
        print(f"[Camera] OpenCV VideoCapture backend 사용 중 (device={device_index})")

    # ----------------- public 메서드 ----------------- #

    def get_frame(self):
        """
        BGR 형식의 프레임을 반환.
        실패 시 None 반환.
        """
        if self.use_picamera2 and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            except Exception as e:
                print(f"[Camera] Picamera2 캡처 오류: {e}")
                return None

        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

        return None

    def release(self):
        if self.use_picamera2 and self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

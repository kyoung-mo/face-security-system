from dataclasses import dataclass

@dataclass
class GPIOConfig:
    enabled: bool
    green_led_pin: int
    red_led_pin: int
    buzzer_pin: int

class GPIOController:
    def __init__(self, config: GPIOConfig):
        self.config = config
        self._setup_done = False

        if self.config.enabled:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.GPIO.setwarnings(False)
                self.GPIO.setmode(GPIO.BCM)
                self.GPIO.setup(self.config.green_led_pin, GPIO.OUT)
                self.GPIO.setup(self.config.red_led_pin, GPIO.OUT)
                self.GPIO.setup(self.config.buzzer_pin, GPIO.OUT)
                self._setup_done = True
            except ImportError:
                print("[GPIO] RPi.GPIO 모듈을 찾을 수 없습니다. GPIO 비활성화.")
                self.config.enabled = False

    def _safe_output(self, pin, value):
        if self.config.enabled and self._setup_done:
            self.GPIO.output(pin, value)
        else:
            # 개발 환경에서는 그냥 print로 대체
            pass

    def green_on(self):
        self._safe_output(self.config.green_led_pin, 1)

    def green_off(self):
        self._safe_output(self.config.green_led_pin, 0)

    def red_on(self):
        self._safe_output(self.config.red_led_pin, 1)

    def red_off(self):
        self._safe_output(self.config.red_led_pin, 0)

    def buzzer_on(self):
        self._safe_output(self.config.buzzer_pin, 1)

    def buzzer_off(self):
        self._safe_output(self.config.buzzer_pin, 0)

    def reset(self):
        self.green_off()
        self.red_off()
        self.buzzer_off()

    def cleanup(self):
        if self.config.enabled and self._setup_done:
            self.GPIO.cleanup()

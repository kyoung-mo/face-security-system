class LCDDisplay:
    def __init__(self):
        # TODO: 실제 사용하는 LCD 라이브러리로 교체
        # 예: luma.lcd, waveshare, ST7735 등
        self.enabled = False

    def show_text(self, text: str):
        if self.enabled:
            # 실제 LCD 출력 코드
            pass
        else:
            # 개발 환경에서는 그냥 출력
            print(f"[LCD] {text}")

# src/lcd_display.py
from RPLCD.i2c import CharLCD


class LCDDisplay:
    def __init__(self):
        # 디버그용 로그
        print("[LCD] Initializing CharLCD (PCF8574, addr=0x27, 16x2)...")

        # ✅ lcd_rplcd_test.py에서 사용해 성공한 설정과 동일하게 맞추기
        self.lcd = CharLCD(
            i2c_expander='PCF8574',
            address=0x27,
            port=1,      # I2C-1 버스
            cols=16,
            rows=2,
            charmap='A00',
            auto_linebreaks=True,
        )

        # 혹시 전에 쓰레기 문자 남아 있을까 봐 초기 클리어
        self.lcd.clear()

    def show_text(self, text: str):
        # 콘솔 로그
        print(f"[LCD] {text}")

        # 🔹 최대 2줄까지 처리 (줄바꿈 기준)
        if "\n" in text:
            line1, line2 = text.split("\n", 1)
        else:
            line1, line2 = text, ""

        # LCD 지우고 커서 위치해서 각각 출력
        self.lcd.clear()
        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(line1[:16])   # 1줄 최대 16자

        if line2:
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(line2[:16])

    def clear(self):
        print("[LCD] clear()")
        self.lcd.clear()

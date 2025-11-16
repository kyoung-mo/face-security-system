# src/lcd_test.py
from lcd_display import LCDDisplay
import time

if __name__ == "__main__":
    lcd = LCDDisplay()  # 이때 [LCD] Initializing ... 로그가 떠야 함

    lcd.show_text("Hello from src")
    time.sleep(2)

    lcd.show_text("LCD from src OK")
    time.sleep(2)

    lcd.clear()

from lcd_display import LCDDisplay
import time

lcd = LCDDisplay()

lcd.show_text("Hello LCD!")
time.sleep(2)

lcd.show_text("Face System OK")
time.sleep(2)

lcd.clear()

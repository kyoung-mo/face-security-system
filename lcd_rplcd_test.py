from RPLCD.i2c import CharLCD
import time

# PCF8574T + 주소 0x27 + 16x2
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,
    port=1,           # I2C-1 버스
    cols=16,
    rows=2,
    charmap='A00',
    auto_linebreaks=True,
)

lcd.clear()
lcd.write_string('Hello LCD! :)')
time.sleep(3)

lcd.clear()
lcd.write_string('Face System OK')
time.sleep(3)

lcd.clear()

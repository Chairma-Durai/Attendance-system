from Adafruit_CharLCD import Adafruit_CharLCD
from time import sleep

# instantiate lcd and specify pins
lcd = Adafruit_CharLCD(rs=21, en=24, d4=23, d5=17, d6=18, d7=22, cols=16, lines=2)
lcd.clear()
# display text on LCD,  \n = new line
lcd.message("Enter a text:")
text=input()
lcd.clear()
lcd.message(text)
sleep(1)

# scroll text on display
while 1:
    for x in range(0, 16):
        lcd.move_left()
        sleep(0.5)
    sleep(2)
    
    # scroll Right
    for x in range(0, 16):
        lcd.move_right()
        sleep(0.5)
    sleep(3)


# Clear the screen
lcd.clear()
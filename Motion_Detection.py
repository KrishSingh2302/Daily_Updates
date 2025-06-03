import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from picamera2 import Picamera2
from datetime import datetime

# Connecting I2C bus and ADS1115 ADC
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)

# threshold above it motion will be detected 
THRESHOLD = 60

# Initialize the camera
camera = Picamera2()
camera.start()

while True:
    value = chan.value
    print("ADC Value:", value)
    if value > THRESHOLD:
        print("Motion detected!")
        # Creating a file name according to current time and date
        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
        filename = f"/home/pi/motion_{timestamp}.jpg"
        camera.capture_file(filename)
        print(f"Photo saved as {filename}")
        time.sleep(1)  # to prevent multiple counting
    time.sleep(0.1)

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from picamera2 import Picamera2
from datetime import datetime
import sqlite3
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import smbus 

# ------------------------- SQL database -------------------------
conn = sqlite3.connect('/home/pi/motion_events.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_path TEXT,
        label TEXT,
        confidence REAL,
        distance_mm INTEGER
    )
''')
conn.commit()

def save_event(timestamp, image_path, label, confidence, distance):
    c.execute("INSERT INTO events (timestamp, image_path, label, confidence, distance_mm) VALUES (?, ?, ?, ?, ?)", 
             (timestamp, image_path, label, confidence, distance))
    conn.commit()

# ------------------------- ML Model Setup -------------------------
model = EfficientNetB0(weights='imagenet')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=1)[0][0]
    return decoded[1], float(decoded[2])

# ------------------------- Hardware Initialization -------------------------
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)

THRESHOLD = 60

camera = Picamera2()
camera.start()

# ------------------------- TOF10120 Sensor Setup -------------------------
bus = smbus.SMBus(1)
tof_address = 0x52  # Confirmed from i2cdetect

def read_tof_distance():
    try:
        high = bus.read_byte_data(tof_address, 0x00)
        low = bus.read_byte_data(tof_address, 0x01)
        return (high << 8) + low
    except Exception as e:
        print("TOF Sensor Error:", e)
        return -1

# ------------------------- Main Loop -------------------------
while True:
    value = chan.value
    print("ADC Value:", value) 
    if value > THRESHOLD:
        print("Motion detected!")
        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
        filename = f"/home/pi/motion_{timestamp}.jpg"
        camera.capture_file(filename)
        print(f"Photo saved as {filename}")

        label, confidence = classify_image(filename)
        print(f"Identified: {label} (confidence: {confidence:.2f})")

        distance = read_tof_distance()
        print(f"TOF Distance: {distance} mm")

        save_event(timestamp, filename, label, confidence, distance)
        print(f"Event saved to database at {timestamp}")
        
        time.sleep(1)
    
    time.sleep(0.1)


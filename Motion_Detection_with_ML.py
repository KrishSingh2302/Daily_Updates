import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from picamera2 import Picamera2
from datetime import datetime
import sqlite3
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import smbus  # For TOF10120 I2C communication

# ------------------------- SQL database -------------------------
# Connect to (or create) the SQLite database for logging events
conn = sqlite3.connect('/home/pi/motion_events.db')
c = conn.cursor()
# Create a table to store event details, including ML results and TOF distance
c.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    image_path TEXT,
    label TEXT,
    confidence REAL,
    distance_mm INTEGER
)
""")
conn.commit()

# Function to save an event to the database
# Stores timestamp, image path, ML label, confidence, and TOF distance
def save_event(timestamp, image_path, label, confidence, distance):
    c.execute("INSERT INTO events (timestamp, image_path, label, confidence, distance_mm) VALUES (?, ?, ?, ?, ?)", 
             (timestamp, image_path, label, confidence, distance))
    conn.commit()

# ------------------------- ML Model Setup -------------------------
# Load the pre-trained MobileNetV2 model with ImageNet weights
model = MobileNetV2(weights='imagenet')

# Function to classify an image and return the label and confidence
def classify_image(img_path):
    # Load the image and resize to 224x224 pixels (required by MobileNetV2)
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to a NumPy array
    x = image.img_to_array(img)
    # Add a batch dimension for ML model input
    x = np.expand_dims(x, axis=0)
    # Preprocess the image (scaling, normalization) for MobileNetV2
    x = preprocess_input(x)
    # Predict the class probabilities for the image
    preds = model.predict(x)
    # Decode the prediction to get the top label and confidence
    decoded = decode_predictions(preds, top=1)[0][0]
    # Return label and confidence
    return decoded[1], float(decoded[2])

# ------------------------- Hardware Initialization -------------------------
# Initialize I2C bus for communication with ADS1115 and TOF10120
i2c = busio.I2C(board.SCL, board.SDA)  # For CircuitPython devices
ads = ADS.ADS1115(i2c)                 # Initialize ADS1115 ADC
chan = AnalogIn(ads, ADS.P0)           # Use channel 0 for CDM324 sensor

# Threshold value for motion detection (ADC reading)
THRESHOLD = 60

# Initialize the Pi Camera
camera = Picamera2()
camera.start()

# TOF10120 Sensor Setup (using smbus for low-level I2C)
bus = smbus.SMBus(1)         # Use I2C bus 1 for TOF10120
tof_address = 0x52           # I2C address of TOF10120 (from i2cdetect)

# Function to read distance from TOF10120 sensor
# Returns distance in millimeters or -1 on error
def read_tof_distance():
    try:
        # Read high byte from register 0x00
        high = bus.read_byte_data(tof_address, 0x00)
        # Read low byte from register 0x01
        low = bus.read_byte_data(tof_address, 0x01)
        # Combine high and low bytes to get distance in mm
        return (high << 8) + low
    except Exception as e:
        print("TOF Sensor Error:", e)
        return -1

# ------------------------- Main Loop -------------------------
while True:
    # Read ADC value from CDM324 sensor
    value = chan.value
    print("ADC Value:", value) 
    # Check if motion is detected (ADC value above threshold)
    if value > THRESHOLD:
        print("Motion detected!")
        # Generate timestamp string for filenames and database
        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
        # Create filename for captured image
        filename = f"/home/pi/motion_{timestamp}.jpg"
        # Capture image from camera and save to file
        camera.capture_file(filename)
        print(f"Photo saved as {filename}")

        # Classify the captured image using ML model
        label, confidence = classify_image(filename)
        print(f"Identified: {label} (confidence: {confidence:.2f})")

        # Read distance from TOF10120 sensor
        distance = read_tof_distance()
        print(f"TOF Distance: {distance} mm")

        # Save event details to database
        save_event(timestamp, filename, label, confidence, distance)
        print(f"Event saved to database at {timestamp}")
        
        # Sleep to prevent multiple detections for the same event
        time.sleep(1)
    
    # Delay to reduce CPU usage
    time.sleep(0.1)

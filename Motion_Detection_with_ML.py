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

# ------------------------- SQL database -------------------------
# Connect to (or create) the SQLite database for logging events
conn = sqlite3.connect('/home/pi/motion_events.db')
c = conn.cursor()
# Create a table to store event details, including ML results
c.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_path TEXT,
        label TEXT,
        confidence REAL
    )
''')
conn.commit()

# Function to save an event to the database
def save_event(timestamp, image_path, label, confidence):
    c.execute("INSERT INTO events (timestamp, image_path, label, confidence) VALUES (?, ?, ?, ?)", 
             (timestamp, image_path, label, confidence))
    conn.commit()

# ------------------------- ML Model Setup -------------------------
# Load the pre-trained MobileNetV2 model with ImageNet weights.
model = MobileNetV2(weights='imagenet')

# Function to classify an image and return the label and confidence
def classify_image(img_path):
    # Load the image and resize to 224x224 pixels (required by MobileNetV2)
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to a NumPy array
    x = image.img_to_array(img)
    # Add a batch dimension for ML model
    x = np.expand_dims(x, axis=0)
    # Preprocess the image for ML Model (eg: scaling, normalization)
    x = preprocess_input(x)
    # Predict the class probabilities for the image
    preds = model.predict(x)
    # Decode the prediction to get the top label and confidence
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]         # Name of the object that it detected
    confidence = float(decoded[2])  # Confidence of getting the object correct
    return label, confidence

# ------------------------- Hardware Initialization -------------------------
# Set up I2C and ADC for the CDM324 motion sensor
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)

THRESHOLD = 60  # Motion detection threshold (adjust as needed)

# Initialize the Pi Camera
camera = Picamera2()
camera.start()

while True:
    value = chan.value
    print("ADC Value:", value) 
    if value > THRESHOLD:
        print("Motion detected!")
        # Create a timestamped filename for the captured image
        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
        filename = f"/home/pi/motion_{timestamp}.jpg"
        camera.capture_file(filename)
        print(f"Photo saved as {filename}")

        # Classify the captured image using ML Model
        label, confidence = classify_image(filename)
        print(f"Identified: {label} (confidence: {confidence:.2f})")

        # Save the event to the database
        save_event(timestamp, filename, label, confidence)
        print(f"Event saved to database at {timestamp}")
        
        time.sleep(1)  # Prevent multiple detection of same object
    
    time.sleep(0.1) 


import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from picamera2 import Picamera2
from datetime import datetime
import sqlite3
import numpy as np

# ------------------------- SQL database -------------------------
# Creates a data
conn = sqlite3.connect('/home/pi/motion_events.db')  # connector to data base
c = conn.cursor()  # Cursor of the data base

# Making tables in the database, if already exists append that table
c.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_path TEXT,
        estimated_distance REAL
    )
''')

# This function saves the time and image path to the database
def save_event(timestamp, image_path, distance):
    c.execute("INSERT INTO events (timestamp, image_path, estimated_distance) VALUES (?, ?, ?)", 
             (timestamp, image_path, distance))
    conn.commit()
conn.commit()

# ------------------------- Doppler Configuration -------------------------
F0 = 24.125e9  # CDM324 frequency (24.125 GHz)
C = 3e8        # Speed of light (m/s)
SAMPLE_RATE = 100  # Hz
BUFFER_SIZE = 100  # Samples for FFT analysis

# ------------------------- Hardware Initialization -------------------------
# Connecting I2C bus and ADS1115 ADC
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)

# Initialize the camera
camera = Picamera2()
camera.start()

# ------------------------- Doppler Processing Variables -------------------------
buffer = []
total_distance = 0.0
prev_time = time.time()

# Threshold above which motion will be detected 
THRESHOLD = 60


while True:
    # Read sensor and update buffer
    current_time = time.time()
    adc_voltage = chan.voltage
    buffer.append(adc_voltage)
    
    # Process FFT every BUFFER_SIZE samples
    if len(buffer) >= BUFFER_SIZE:
        dt = current_time - prev_time
        prev_time = current_time
        
        # Calculate Doppler frequency shift using FFT
        fft = np.fft.rfft(buffer)
        freqs = np.fft.rfftfreq(len(buffer), 1/SAMPLE_RATE)
        dominant_freq = freqs[np.argmax(np.abs(fft))]
        buffer = []  # Reset buffer

        # Calculate speed (ignore negative values)
        speed = (abs(dominant_freq) * C) / (2 * F0)
        total_distance += speed * dt

    # Motion detection logic
    value = chan.value
    print("ADC Value:", value)	
    if value > THRESHOLD:
        print("Motion detected!")
        print(f"Current estimated distance: {total_distance:.2f} meters")
        
        # Creating a file name according to current time and date
        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")  # Original timestamp format
        filename = f"/home/pi/motion_{timestamp}.jpg"
        camera.capture_file(filename)
        print(f"Photo saved as {filename}")
        
        # Save event to database with distance
        save_event(timestamp, filename, total_distance)
        print(f"Event saved to database at {timestamp}")
        
        # Reset distance for new detection
        total_distance = 0.0
        time.sleep(1)  # to prevent multiple counting
    
    time.sleep(1/SAMPLE_RATE)  # Maintain consistent sampling rate

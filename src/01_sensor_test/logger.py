import serial
import csv
import time
import os
from datetime import datetime
# --- Parameters ---
duration = 120

# --- Config ---
PORT = "COM3"
BAUD_RATE = 9600       # Must match Serial.begin() in your sketch
os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data\\data.csv"
INFO = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Type: Fixed-time Logger", f"Duration: {duration}s"]
HEADERS = ["time_s", "force_N", "weight_g"]

def log_serial(port, baud, output_file, info, headers, duration_seconds):
    print(f"Logging to {output_file} for {duration_seconds}s... (Ctrl+C to stop early)")
    with serial.Serial(port, baud, timeout=1) as ser, \
         open(output_file, "w", newline="") as f:
        
        writer = csv.writer(f)
        writer.writerow(INFO)
        writer.writerow(HEADERS)
        
        start = time.time()
        ser.reset_input_buffer()  # flush stale data

        while time.time() - start < duration_seconds:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            
            values = line.split(",")
            # Replace timestamp with relative time in seconds from logging start
            values[0] = (time.time() - start)
            if len(values) == len(headers):
                writer.writerow(values)
                f.flush()  # write immediately, don't buffer
                print(values)  # live preview

if __name__ == "__main__":
    log_serial(PORT, BAUD_RATE, OUTPUT_FILE, INFO, HEADERS, duration)
    print("Done!")
import time
import serial
from config import PORT, BAUD_RATE

class Sensor:
    def __init__(self):
        self.ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # wait for Arduino to reset after connection
        self.ser.reset_input_buffer()

    def readline(self) -> str:
        return self.ser.readline().decode("utf-8", errors="replace").strip()

    def send(self, msg: str):
        self.ser.write((msg + "\n").encode())

    def parse(self, line: str):
        """Parse 'time_s,force_N,weight_g'. Returns tuple or None on failure."""
        try:
            parts = line.split(",")
            if len(parts) == 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            pass
        return None

    def close(self):
        self.ser.close()
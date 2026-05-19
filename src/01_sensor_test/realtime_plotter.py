import csv
import os
from datetime import datetime
from collections import deque


import matplotlib.pyplot as plt
import serial

PORT = "COM3"
BAUD_RATE = 9600
MAX_POINTS = 200  # how many points to keep on the screen
os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data\\data.csv"
INFO = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Type: Real-time Plotter"]
HEADERS = ["time_s", "force_N", "weight_g"]



def parse_line(line):
    try:
        time, force, weight = line.split(",")
        return float(time), float(force), float(weight)
    except ValueError:
        return None


def main():
    print(f"Opening serial port {PORT} at {BAUD_RATE} bps")
    with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser, open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(INFO)
        writer.writerow(HEADERS)

        x = deque(maxlen=MAX_POINTS)
        y_force = deque(maxlen=MAX_POINTS)
        y_weight = deque(maxlen=MAX_POINTS)

        plt.ion()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        line_force, = ax1.plot([], [], color="tomato", marker="o", markersize=4, linestyle="-", linewidth=1, label="Force (N)")
        line_weight, = ax2.plot([], [], color="steelblue", marker="o", markersize=4, linestyle="-", linewidth=1, label="Weight (g)")
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Force (N)", color="tomato")
        ax1.tick_params(axis="y", labelcolor="tomato")
        ax1.set_title("Real-time Arduino Force and Weight Data")
        ax1.grid(True)
        
        ax2.set_ylabel("Weight (g)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        
        lines = [line_force, line_weight]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        start_time = None

        print("Waiting for data...")
        while True:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                plt.pause(0.01)
                continue

            parsed = parse_line(raw)
            if parsed is None:
                continue

            time, force, weight = parsed
            if start_time is None:
                start_time = time

            time_s = (time - start_time)
            x.append(time_s)
            y_force.append(force)
            y_weight.append(weight)

            writer.writerow([time_s, force, weight])
            f.flush()

            line_force.set_data(x, y_force)
            line_weight.set_data(x, y_weight)
            
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(f"{time_s:.2f}s -> Force: {force:.2f}N, Weight: {weight:.2f}g")

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user")

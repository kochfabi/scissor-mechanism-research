import os
import time
import threading
from sensor import Sensor
from config import CAPTURE_DURATION_S, OUTPUT_DIR
from analysis import compute_stats, save_results, plot_results


# ── Calibration ──────────────────────────────────────────────────────────────

def run_calibration(sensor: Sensor):
    def wait_for(prefix: str):
        while True:
            line = sensor.readline()
            if not line:
                continue
            if line.startswith(prefix):
                print("  ", line)
                return line
    
    print("\n=== CALIBRATION ===")
    input("  Remove all weight from the load cell, then press Enter...")
    sensor.send("")  # sends '\n' — triggers Arduino tare
    wait_for("OFFSET")

    weight_g = input("  Place known calibration weight. Enter mass [g]: ").strip()
    sensor.send(weight_g)
    wait_for("SCALE")

    while True:
        line = sensor.readline()
        if line == "READY":
            print("  Sensor ready.\n")
            break


# ── Experiment setup ─────────────────────────────────────────────────────────

def get_experiment_metadata() -> dict:
    print("=== EXPERIMENT SETUP ===")
    variable = input("  Independent variable (e.g. F_in, l_offset, n_units, l_curve): ").strip()
    unit     = input(f"  Unit for '{variable}': ").strip()
    notes    = input("  Notes (Enter to skip): ").strip()
    return {"variable": variable, "unit": unit, "notes": notes}


def parse_float(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def compute_efficiency(input_value: float, input_unit: str, stats: dict):
    if input_value is None or input_value == 0:
        return None

    unit = input_unit.strip().lower()
    if unit in ("n", "N", "newton", "newtons"):
        return stats["mean_force_N"] / input_value
    if unit in ("g", "gram", "grams"):
        return stats["mean_force_g"] / input_value
    if unit in ("kg", "kilogram", "kilograms"):
        return stats["mean_force_g"] / (input_value * 1000.0)
    return stats["mean_force_g"] / input_value


# ── Live display ─────────────────────────────────────────────────────────────

def show_live_until_enter(sensor: Sensor):
    """Stream live readings to terminal. Returns when user presses Enter."""
    print("  Live readings (press Enter when force is stable):")

    stop = threading.Event()

    def read_loop():
        while not stop.is_set():
            line = sensor.readline()
            parsed = sensor.parse(line)
            if parsed:
                _, _, w = parsed
                print(f"    {w:+.4f} g", end="\r")

    thread = threading.Thread(target=read_loop, daemon=True)
    thread.start()
    input()          # blocks until Enter
    stop.set()
    thread.join(timeout=1)
    print()          # newline after the \r output


# ── Capture ───────────────────────────────────────────────────────────────────

def capture(sensor: Sensor) -> list:
    """Record CAPTURE_DURATION_S seconds of data after Enter is pressed."""
    print(f"  Capturing {CAPTURE_DURATION_S}s...", end="", flush=True)
    deadline = time.time() + CAPTURE_DURATION_S
    readings = []
    while time.time() < deadline:
        parsed = sensor.parse(sensor.readline())
        if parsed:
            readings.append(parsed)
    print(f" done. ({len(readings)} samples)")
    return readings


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sensor = Sensor()
    run_calibration(sensor)
    metadata = get_experiment_metadata()

    variable = metadata["variable"]
    trials   = []
    print(f"\n=== MEASUREMENT — varying {variable} ===")
    print("  For each trial: configure your setup, watch live readings, press Enter to capture.")
    print("  Type 'done' as the condition value to finish.\n")

    trial_n = 1
    while True:
        print(f"--- Trial {trial_n} ---")
        value = input(f"  {variable} = ").strip()
        if value.lower() == "done":
            break

        show_live_until_enter(sensor)
        readings = capture(sensor)
        stats    = compute_stats(readings)
        input_value = parse_float(value)
        epsilon = compute_efficiency(input_value, metadata["unit"], stats)

        if epsilon is None:
            print(f"  → {stats['mean_force_g']:+.6f} ± {stats['std_force_g']:.6f} g  (n = {stats['n']})\n")
        else:
            print(f"  → {stats['mean_force_g']:+.6f} ± {stats['std_force_g']:.6f} g  (ε = {epsilon:.3f}, {epsilon * 100:.1f}%)  (n = {stats['n']})\n")

        trials.append({
            "trial":         trial_n,
            variable:        value,
            "mean_force_g":  stats["mean_force_g"],
            "std_force_g":   stats["std_force_g"],
            "epsilon":       epsilon,
            "n_samples":     stats["n"],
            "raw":           readings,
        })
        trial_n += 1

    if not trials:
        print("No trials recorded.")
        sensor.close()
        return

    exp_dir = save_results(trials, metadata)
    plot_results(trials, metadata, exp_dir)
    sensor.close()


if __name__ == "__main__":
    main()
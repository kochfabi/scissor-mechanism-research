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

    # Tare after mounting to mechanism
    input("  Remove calibration weight, mount sensor, then press Enter...")
    show_live_until_enter(sensor, "Press Enter to tare...")
    sensor.send("")  # sends '\n' — triggers Arduino tare
    wait_for("READY")

# ── Experiment setup ─────────────────────────────────────────────────────────

def get_experiment_metadata() -> dict:
    print("=== EXPERIMENT SETUP ===")
    title = input("  Experiment title (Enter to skip): ").strip()
    title = title if title != "" else "Untitled"
    notes = input("  Notes (Enter to skip): ").strip()
    
    variable_map = {
        "f_in": "F_in",
        "l_offset": "l_offset",
        "n_units": "n_units",
        "l_curve": "l_curve",
    }
    while True:
        choice = input(
            "  Independent variable (F_in, l_offset, n_units, l_curve): "
        ).strip().lower()
        independent_variable = variable_map.get(choice)
        if independent_variable:
            break
        print(f"  Invalid choice. Choose one of: {', '.join(variable_map.values())}")

    independent_unit = input(f"  Unit for '{independent_variable}': ").strip()

    if independent_variable == "F_in":
        F_in = ""
        F_in_unit = independent_unit
    else:
        F_in = input("  F_in (input force): ").strip()
        F_in_unit = input("  Unit for F_in: ").strip()
        
    l_offset = "" if independent_variable == "l_offset" else input("  l_offset (opening angle offset length): ").strip()
    n_units = "" if independent_variable == "n_units" else input("  n_units (number of units): ").strip()
    l_curve = "" if independent_variable == "l_curve" else input("  l_curve (curvilinear offset length): ").strip()

    return {
        "independent_variable": independent_variable,
        "independent_unit": independent_unit,
        "F_in": F_in,
        "F_in_unit": F_in_unit,
        "l_offset": l_offset,
        "n_units": n_units,
        "l_curve": l_curve,
        "notes": notes,
        "title": title,
    }


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

def show_live_until_enter(
    sensor: Sensor,
    prompt: str = "Live readings (press Enter when force is stable):"
):
    """Stream live readings to a live plot.
    Auto-continues when std dev < 0.3 g for last 50 samples
    after a minimum wait time of 15 s.
    """

    print(f"  {prompt}")

    import matplotlib.pyplot as plt
    import numpy as np
    from collections import deque

    window_size = 50
    stability_threshold = 0.3  # g
    minimum_wait_s = 15

    timestamps = deque(maxlen=window_size)
    weights = deque(maxlen=window_size)

    stop = threading.Event()
    enter_pressed = threading.Event()

    start_time = time.time()

    def input_thread():
        input()
        enter_pressed.set()

    def read_loop():
        while not stop.is_set():
            line = sensor.readline()
            parsed = sensor.parse(line)

            if parsed:
                _, _, w = parsed
                timestamps.append(time.time() - start_time)
                weights.append(w)
            else:
                time.sleep(0.01)

    input_t = threading.Thread(target=input_thread, daemon=True)
    input_t.start()

    read_t = threading.Thread(target=read_loop, daemon=True)
    read_t.start()

    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 6))

    line_plot, = ax.plot(
        [],
        [],
        color="steelblue",
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1,
        label="Weight (g)"
    )

    avg_line = ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        label="Mean"
    )

    stats_text = ax.text(
        0.02,
        0.20,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_title(prompt)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight (g)")
    ax.grid(True)
    ax.legend(loc="upper left")

    try:
        while not enter_pressed.is_set():

            if weights:

                weights_np = np.array(weights)

                current_value = weights_np[-1]
                mean_value = np.mean(weights_np)
                std_value = np.std(weights_np)

                line_plot.set_data(timestamps, weights)
                avg_line.set_ydata([mean_value, mean_value])

                stats_text.set_text(
                    f"Current: {current_value:+.3f} g\n"
                    f"Mean: {mean_value:+.3f} g\n"
                    f"Std Dev: {std_value:.3f} g\n"
                    f"Samples: {len(weights_np)}"
                )

                ax.relim()
                ax.autoscale_view()

                if len(timestamps) > 1:
                    ax.set_xlim(timestamps[0], timestamps[-1])

                fig.canvas.draw()
                fig.canvas.flush_events()

                elapsed = time.time() - start_time

                if (
                    elapsed >= minimum_wait_s
                    and len(weights_np) >= window_size
                    and std_value < stability_threshold
                ):
                    enter_pressed.set()

            plt.pause(0.05)

    except Exception:
        pass

    finally:
        stop.set()
        read_t.join(timeout=1)

        plt.ioff()
        plt.close(fig)

        print()


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
    metadata = get_experiment_metadata()
    run_calibration(sensor)

    independent_variable = metadata["independent_variable"]
    trials   = []
    print(f"\n=== MEASUREMENT — varying {independent_variable} ===")
    print("  For each trial: configure your setup, watch live readings, press Enter to capture.")
    print("  Type 'done' as the condition value to finish.\n")

    trial_n = 1
    while True:
        print(f"--- Trial {trial_n} ---")
        value = input(f"  {independent_variable} = ").strip()
        if value.lower() == "done":
            break

        show_live_until_enter(sensor)
        readings = capture(sensor)
        stats    = compute_stats(readings)

        if independent_variable == "F_in":
            input_value = parse_float(value)
            efficiency_unit = metadata["independent_unit"]
        else:
            input_value = parse_float(metadata["F_in"])
            efficiency_unit = metadata["F_in_unit"]

        epsilon = compute_efficiency(input_value, efficiency_unit, stats)

        if epsilon is None:
            print(f"  → {stats['mean_force_g']:+.6f} ± {stats['std_force_g']:.6f} g  (n = {stats['n']})\n")
        else:
            print(f"  → {stats['mean_force_g']:+.6f} ± {stats['std_force_g']:.6f} g  (ε = {epsilon:.3f})  (n = {stats['n']})\n")

        trials.append({
            "trial":         trial_n,
            independent_variable: value,
            "mean_force_g":  stats["mean_force_g"],
            "std_force_g":   stats["std_force_g"],
            "epsilon":       epsilon,
            "n_samples":     stats["n"],
            "raw":           readings,
            "drift_slope":   stats["drift_slope"]
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
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import OUTPUT_DIR


def compute_stats(readings: list) -> dict:
    forces = [r[1] for r in readings]
    return {
        "mean": float(np.mean(forces)),
        "std":  float(np.std(forces, ddof=1)),  # sample std (Bessel's correction)
        "n":    len(forces),
    }


def save_results(trials: list, metadata: dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir   = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    variable  = metadata["variable"]

    # Summary table
    with open(os.path.join(exp_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# variable", variable])
        w.writerow(["# unit",     metadata["unit"]])
        w.writerow(["# notes",    metadata["notes"]])
        w.writerow(["trial", variable, "mean_force_N", "std_force_N", "n_samples"])
        for t in trials:
            w.writerow([t["trial"], t[variable],
                        t["mean_force_N"], t["std_force_N"], t["n_samples"]])

    # Raw data per trial
    for t in trials:
        path = os.path.join(exp_dir, f"trial_{t['trial']:02d}_raw.csv")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["time_s", "force_N", "weight_g"]] + list(t["raw"]))

    print(f"\nData saved → {exp_dir}")
    return exp_dir


def plot_results(trials: list, metadata: dict):
    variable = metadata["variable"]
    unit     = metadata["unit"]
    x_labels = [t[variable] for t in trials]
    means    = [t["mean_force_N"] for t in trials]
    stds     = [t["std_force_N"]  for t in trials]

    try:
        x = [float(v) for v in x_labels]
        numeric = True
    except ValueError:
        x = list(range(len(trials)))
        numeric = False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, means, yerr=stds, fmt="o-", color="black",
                ecolor="gray", elinewidth=1.5, capsize=5, label="Mean ± std (sample)")
    ax.set_xlabel(f"{variable} [{unit}]" if unit else variable)
    ax.set_ylabel("Output Force (N)")
    ax.set_title(f"Output Force vs {variable}")
    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    notes = metadata.get("notes", "")
    if notes:
        fig.text(0.99, 0.01, f"Notes: {notes}",
                 ha="right", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"plot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved  → {path}")
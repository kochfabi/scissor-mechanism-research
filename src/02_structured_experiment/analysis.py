import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import OUTPUT_DIR


def compute_stats(readings: list) -> dict:
    # readings: (time_s, force_N, weight_g)
    forces_N = [r[1] for r in readings]
    forces_g = [r[2] for r in readings]
    return {
        "mean_force_g": float(np.mean(forces_g)),
        "std_force_g":  float(np.std(forces_g, ddof=1)),  # sample std (Bessel's correction)
        "mean_force_N": float(np.mean(forces_N)),
        "std_force_N":  float(np.std(forces_N, ddof=1)),
        "n":            len(forces_g),
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
        # Summary uses mass-equivalent in grams for readability and reports efficiency.
        w.writerow(["trial", variable, "mean_force_g", "std_force_g", "epsilon", "epsilon_%", "n_samples"])
        for t in trials:
            epsilon = t.get("epsilon")
            epsilon_pct = f"{epsilon * 100:.2f}" if epsilon is not None else ""
            w.writerow([t["trial"], t[variable],
                        t["mean_force_g"], t["std_force_g"],
                        epsilon if epsilon is not None else "",
                        epsilon_pct,
                        t["n_samples"]])

    # Raw data per trial
    for t in trials:
        path = os.path.join(exp_dir, f"trial_{t['trial']:02d}_raw.csv")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["time_s", "force_N", "weight_g"]] + list(t["raw"]))

    print(f"\nData saved → {exp_dir}")
    return exp_dir


def plot_results(trials: list, metadata: dict, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    variable = metadata["variable"]
    unit     = metadata["unit"]
    x_labels = [t[variable] for t in trials]
    means    = [t["mean_force_g"] for t in trials]
    stds     = [t["std_force_g"]  for t in trials]
    epsilons = [t.get("epsilon") for t in trials]
    has_epsilon = any(e is not None for e in epsilons)

    try:
        x = [float(v) for v in x_labels]
        numeric = True
    except ValueError:
        x = list(range(len(trials)))
        numeric = False

    fig, ax = plt.subplots(figsize=(8, 5))
    error_line = ax.errorbar(x, means, yerr=stds, fmt="o-", color="black",
                             ecolor="gray", elinewidth=1.5, capsize=5, label="Output force (g) ± std")
    ax.set_xlabel(f"{variable} [{unit}]" if unit else variable)
    ax.set_ylabel("Output Force (g)")
    ax.set_title(f"Output Force vs {variable}")
    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle="--", alpha=0.5)

    lines = [error_line[0]]
    labels = [error_line[0].get_label()]
    if has_epsilon:
        ax2 = ax.twinx()
        eps_plot = [100 * e if e is not None else np.nan for e in epsilons]
        eps_line, = ax2.plot(x, eps_plot, color="tab:blue", marker="s", linestyle="-", linewidth=1.5,
                             label="Efficiency (ε %)")
        ax2.set_ylabel("Efficiency ε (%)")
        ax2.grid(False)
        lines.append(eps_line)
        labels.append(eps_line.get_label())

    ax.legend(lines, labels, loc="upper left")

    notes = metadata.get("notes", "")
    if notes:
        fig.text(0.99, 0.01, f"Notes: {notes}",
                 ha="right", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    path = os.path.join(output_dir, f"plot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved  → {path}")
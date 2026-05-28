import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import OUTPUT_DIR, CAPTURE_DURATION_S


def compute_stats(readings: list) -> dict:
    # readings: (time_s, force_N, weight_g)
    times = [r[0] for r in readings]
    forces_N = [r[1] for r in readings]
    forces_g = [r[2] for r in readings]
    return {
        "mean_force_g": float(np.mean(forces_g)),
        "std_force_g":  float(np.std(forces_g, ddof=1)),  # sample std (Bessel's correction)
        "mean_force_N": float(np.mean(forces_N)),
        "std_force_N":  float(np.std(forces_N, ddof=1)),
        "n":            len(forces_g),
        "drift_slope": float(np.polyfit(times, forces_g, 1)[0]),
        "drift_r": float(np.corrcoef(times, forces_g)[0, 1])
    }


def save_results(trials: list, metadata: dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    title = metadata.get("title")
    exp_dir   = os.path.join(OUTPUT_DIR, timestamp + "_" + title)
    os.makedirs(exp_dir, exist_ok=True)
    variable  = metadata["independent_variable"]

    #  ── Summary table ─────────────────────────────────────────────────
    with open(os.path.join(exp_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# title", metadata.get("title", "")])
        w.writerow(["# notes", metadata.get("notes", "")])
        w.writerow(["# independent_variable", variable + (f" [{metadata['independent_unit']}]" if metadata.get("independent_unit") else "")])
        F_in_unit = metadata.get("F_in_unit", "")
        w.writerow(["# F_in", metadata["F_in"] + (f" {F_in_unit}" if F_in_unit else "")])
        w.writerow(["# l_offset", metadata["l_offset"]])
        w.writerow(["# n_units", metadata["n_units"]])
        w.writerow(["# l_curve", metadata["l_curve"]])
        # Summary uses mass-equivalent in grams for readability and reports efficiency.
        w.writerow(["trial", variable, "mean_force_g", "std_force_g", "epsilon", "n_samples", "drift_slope_g_s"])
        for t in trials:
            epsilon = t.get("epsilon")
            w.writerow([t["trial"], t[variable], t["mean_force_g"], t["std_force_g"],
                        epsilon if epsilon is not None else "", t["n_samples"], t["drift_slope"]])

    #  ── Raw data per trial ─────────────────────────────────────────────────
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

    # ── Summary Plot ─────────────────────────────────────────────────
    variable = metadata["independent_variable"]
    unit     = metadata.get("independent_unit", "")
    x_labels = [t[variable] for t in trials]
    means    = [t["mean_force_g"] for t in trials]
    stds     = [t["std_force_g"] for t in trials]
    epsilons = [t.get("epsilon") for t in trials]
    has_epsilon = any(e is not None for e in epsilons)

    try:
        x = [float(v) for v in x_labels]
        numeric = True
    except ValueError:
        x = list(range(len(trials)))
        numeric = False

    fig, ax = plt.subplots(figsize=(8, 7))
    # ── Force output plot ────────────────────────────────────────────
    ax.errorbar(x, means, yerr=stds, fmt="o-", color="black", markersize=4, linewidth=1, ecolor="gray", elinewidth=1.5, capsize=5, label="F_out [g] ± std")

    # ── Ideal line ───────────────────────────────────────────────────
    ax.plot(x, x, "--", color="red", label="ideal (F_out = F_in)")

    ax.set_xlabel(f"{variable} [{unit}]" if unit else variable)
    ax.set_ylabel("F_out [g]")
    ax.set_title(f"F_out vs {variable}")
    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Efficiency plot ──────────────────────────────────────────────
    if has_epsilon:
        ax2 = ax.twinx()
        eps_plot = [
            100 * e if e is not None else np.nan
            for e in epsilons
        ]
        ax2.plot( x, eps_plot, color="tab:blue", marker="s", linestyle="-", linewidth=1.5, label="Efficiency ε (%)"
        )

        ax2.set_ylabel("Efficiency ε (%)")
        ax2.grid(False)

    # ── Combined legend from both axes ───────────────────────────────

    handles1, labels1 = ax.get_legend_handles_labels()
    if has_epsilon:
        handles2, labels2 = ax2.get_legend_handles_labels()
    else:
        handles2, labels2 = [], []
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left"
    )

    # ── Bottom annotations ───────────────────────────────────────────

    notes = metadata.get("notes", "")

    # Notes (bottom left)
    plot_notes_text = f"Notes: {notes}\n"

    # Variables (bottom right)
    plot_variables = [
        f"independent_variable = {metadata['independent_variable']} [{metadata.get('independent_unit','')}]",
        f"F_in = {metadata.get('F_in','')} [{metadata.get('F_in_unit','')}]",
        f"l_offset = {metadata.get('l_offset','')}",
        f"n_units = {metadata.get('n_units','')}",
        f"l_curve = {metadata.get('l_curve','')}",
        f"capture_duration = {CAPTURE_DURATION_S}s"
    ]

    plot_variables_text = "\n".join(
        x for x in plot_variables
        if x and x != " []"
    )

    # Notes display
    fig.text(0.01, 0.01, plot_notes_text, ha="left", va="bottom", fontsize=9, color="darkgray")
    
    # Variables display
    if plot_variables_text:
        fig.text(0.99, 0.01, plot_variables_text, ha="right", va="bottom", fontsize=9, color="darkgray")

    # ── Layout + save ────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    summary_path = os.path.join(output_dir, "plot.png")
    plt.savefig(summary_path, dpi=150)
    plt.show()
    print(f"Plot saved  → {summary_path}")

    #  ── Individual Trial Plots ─────────────────────────────────────────────────
    for t in trials:
        trial_value = t[variable]
        trial_label = f"{variable} = {trial_value}"
        trial_times = [row[0] for row in t["raw"]]
        trial_weights = [row[2] for row in t["raw"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(trial_times, trial_weights, marker="o", linestyle="-", color="black", label="F_out [g]")
        
        # Add mean line
        mean_force = t["mean_force_g"]
        std_force = t["std_force_g"]
        time_range = [trial_times[0], trial_times[-1]]
        ax.axhline(y=mean_force, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_force:.4f} g")
        
        # Add standard deviation band
        ax.fill_between(time_range, mean_force - std_force, mean_force + std_force, 
                        alpha=0.1, color="red", label=f"Std Dev: ±{std_force:.4f} g")
        
        ax.set_xlabel("time [s]")
        ax.set_ylabel("F_out [g]")
        ax.set_title(f"Trial {t['trial']:02d}: {trial_label}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

        # Notes (bottom left)
        plot_notes_text = f"Notes: {notes}\n"
        
        # Variables (bottom right)
        plot_variables = [
            f"independent_variable = {metadata['independent_variable']} [{metadata.get('independent_unit','')}]",
            f"F_in = {metadata.get('F_in','')} [{metadata.get('F_in_unit','')}]",
            f"l_offset = {metadata.get('l_offset','')}",
            f"n_units = {metadata.get('n_units','')}",
            f"l_curve = {metadata.get('l_curve','')}",
            f"capture_duration = {CAPTURE_DURATION_S}s"
        ]
        plot_variables_text = "\n".join(x for x in plot_variables if x and x != " []")

        # Display notes on bottom left with better contrast
        if notes or True:
            fig.text(0.01, 0.01, plot_notes_text,
                     ha="left", va="bottom", fontsize=9, color="darkgray")
        
        # Display variables on bottom right
        if plot_variables_text:
            fig.text(0.99, 0.01, plot_variables_text,
                     ha="right", va="bottom", fontsize=9, color="darkgray")

        plt.tight_layout(rect=[0, 0.15, 1, 1])
        trial_path = os.path.join(output_dir, f"plot_trial_{t['trial']:02d}.png")
        plt.savefig(trial_path, dpi=600)
        plt.show()
        print(f"Trial plot saved → {trial_path}")
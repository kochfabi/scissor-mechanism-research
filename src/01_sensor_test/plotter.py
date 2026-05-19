import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

df = pd.read_csv("data\\data.csv", skiprows=1)


# --- Basic stats ---
print(df.describe())

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["time_s"], df["weight_g"], color="black", label="Weight", marker="o", markersize=4, markeredgecolor="blue", linestyle="--", linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Weight (g)")
ax.legend()
ax.grid(True)

# --- Weight distribution with error bars ---
num_bins = 1
bins = pd.cut(df["time_s"], bins=num_bins)
weight_stats = df.groupby(bins)["weight_g"].agg(["mean", "std"])
bin_centers = [interval.mid for interval in weight_stats.index]

ax.errorbar(
    bin_centers,
    weight_stats["mean"],
    yerr=weight_stats["std"].fillna(0),
    fmt="o",
    color="red",
    ecolor="gray",
    elinewidth=1.5,
    capsize=4,
    label="Mean weight ± std"
)

ax.set_title("Force Sensor Data")
ax.legend()
ax.grid(True)

# --- Annotate overall statistics ---
mean_weight = df["weight_g"].mean()
std_weight = df["weight_g"].std()
std_force = df["force_N"].std()
stats_text = (
    f"Mean Weight: {mean_weight:.3f} g\n"
    f"Std Weight: {std_weight:.3f} g\n"
    f"Std Force: {std_force * 1000:.3f} mN"
)
ax.text(
    0.98,
    0.97,
    stats_text,
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="right",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", boxstyle="round,pad=0.25")
)

plt.tight_layout()
os.makedirs("data", exist_ok=True)
creation_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f"data\\plot_{creation_date}.png", dpi=150)
plt.show()
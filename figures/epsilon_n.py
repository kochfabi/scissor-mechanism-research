"""
make_plot_epsilon_n.py
-----------------------
Regenerates plot_epsilon_n.png (efficiency epsilon vs. F_in for n=1-3 units)
from metrics.csv, using the consistent publication-quality style defined in
plot_style.py.

Visualizes the loading and unloading branches to highlight the widening
hysteresis gap as the number of units n increases, complete with directional
arrows. The legend combines the load/unload handles into a single row per n.
"""

import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from plot_style import new_fig, COLOR_DATA, COLOR_FIT, COLOR_MODEL, COLOR_IDEAL

SRC = r"C:\Users\fabia\OneDrive - Science Tokyo\Documents\4-Software\data\Analysis\2026-06-10_LinkNumberVariation\metrics.csv"
OUT = r"C:\Users\fabia\OneDrive - Science Tokyo\Documents\5-Presentation\figures/plot_epsilon_n.png"

# ─── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(SRC)

# Filter for the relevant experiment (LinkNumberVariationHysteresis)
df_hysteresis = df[df['label'].str.contains('LinkNumberVariationHysteresis')]

# Sort unique n values to ensure consistent plotting order
ns = sorted(df_hysteresis['n_units'].unique())

# Map each n to a distinct color from plot_style.py and a distinct marker
styles = {
    1: {"color": COLOR_FIT,   "marker": "o"},  # Dark navy
    2: {"color": COLOR_MODEL, "marker": "s"},  # Mid blue
    3: {"color": COLOR_DATA,  "marker": "^"},  # Brick red
}

# ─── Plot ─────────────────────────────────────────────────────────────────
fig, ax = new_fig()

# Plot ideal efficiency line (epsilon = 1)
ideal_line = ax.axhline(1.0, color=COLOR_IDEAL, linestyle=":", linewidth=1.5, zorder=1)

# Prepare lists to hold legend handles and their corresponding labels
legend_handles = [ideal_line]
legend_labels = [r"Ideal ($\epsilon=1$)"]

for n in ns:
    # Filter and sort data for the current n
    df_n = df_hysteresis[df_hysteresis['n_units'] == n].sort_values('F_in_g')
    
    # Extract as numpy arrays to safely iterate via index for the arrows
    f_in = df_n['F_in_g'].values
    eps_load = df_n['epsilon_load'].values
    eps_unload = df_n['epsilon_unload'].values
    
    color = styles[n]["color"]
    marker = styles[n]["marker"]
    
    # ─── Loading Branch ───
    # We assign the plot object to `p_load` (note the comma) to use it in the legend
    p_load, = ax.plot(f_in, eps_load, linestyle="-", marker=marker, color=color,
                      linewidth=1.4, markersize=5, zorder=3)
            
    # Add directional arrows for the loading branch (pointing right/forward)
    for i in range(len(f_in) - 1):
        if pd.isna(eps_load[i]) or pd.isna(eps_load[i+1]):
            continue
        mid_fin = (f_in[i] + f_in[i+1]) / 2
        mid_eps = (eps_load[i] + eps_load[i+1]) / 2
        dx = (f_in[i+1] - f_in[i]) * 0.01
        dy = (eps_load[i+1] - eps_load[i]) * 0.01
        ax.annotate("", xy=(mid_fin + dx, mid_eps + dy), xytext=(mid_fin - dx, mid_eps - dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1, alpha=0.8), zorder=4)

    # ─── Unloading Branch ───
    # We assign the plot object to `p_unload` to use it in the legend
    p_unload, = ax.plot(f_in, eps_unload, linestyle="--", marker=marker, color=color,
                        linewidth=1.4, markersize=5, markerfacecolor="white", zorder=3)
            
    # Add directional arrows for the unloading branch (pointing left/backward)
    for i in range(len(f_in) - 1):
        if pd.isna(eps_unload[i]) or pd.isna(eps_unload[i+1]):
            continue
        mid_fin = (f_in[i] + f_in[i+1]) / 2
        mid_eps = (eps_unload[i] + eps_unload[i+1]) / 2
        dx = (f_in[i+1] - f_in[i]) * 0.01
        dy = (eps_unload[i+1] - eps_unload[i]) * 0.01
        ax.annotate("", xy=(mid_fin - dx, mid_eps - dy), xytext=(mid_fin + dx, mid_eps + dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1, alpha=0.8), zorder=4)

    # Group the two line handles together as a tuple for this n's legend entry
    legend_handles.append((p_load, p_unload))
    legend_labels.append(fr"$n={n}$ (load/unload)")

ax.set_xlabel(r"Input force $F_{in}$ (g)")
ax.set_ylabel(r"Efficiency $\epsilon = F_{out}/F_{in}$")
ax.set_title(r"Efficiency $\epsilon$ vs. $F_{in}$")

# Set sensible limits to match the reference style and frame the data well
ax.set_xlim(50, 550)
ax.set_ylim(0.75, 1.25)

# Render the legend using HandlerTuple so the tuple of lines plots side-by-side
ax.legend(
    legend_handles, 
    legend_labels, 
    loc="lower right", 
    handler_map={tuple: HandlerTuple(ndivide=None)}, 
    fontsize=10
)

fig.savefig(OUT)
print(f"Saved -> {OUT}")
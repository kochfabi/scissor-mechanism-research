"""
plot_style.py
-------------
Shared matplotlib style for all figures in the scissor-gripper paper.
Import and call apply_style() before creating any figure so that every
plot in the paper shares fonts, colors, sizes, and layout conventions.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── Palette ────────────────────────────────────────────────────────────────
# One consistent, colorblind-safe palette reused across all figures.
COLOR_DATA = "#C0392B"      # measured / experimental data      (brick red)
COLOR_FIT = "#1A5276"       # discrete two-stage-fit result     (dark navy)
COLOR_MODEL = "#2E86C1"     # continuous theoretical model      (mid blue)
COLOR_IDEAL = "#7F8C8D"     # ideal / reference line            (neutral gray)
COLOR_TREND = "#7D3C98"     # auxiliary regression / trend line (muted purple)
COLOR_GRID = "#B0B0B0"

FIGSIZE = (5.0, 3.75)       # inches; consistent aspect ratio across figures
DPI = 300


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 14.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.markersize": 5,
        "lines.linewidth": 1.4,
        "errorbar.capsize": 3.5,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.axisbelow": True,
    })


def new_fig():
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    return fig, ax
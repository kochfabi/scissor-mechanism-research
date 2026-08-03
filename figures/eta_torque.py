"""
make_plot_eta_torque.py
------------------------
Regenerates plot_eta_torque.png (per-unit efficiency eta vs. controlled
joint preload torque) from analysis.xlsx, sheet 'fitted_params_per_config'
(the HighFrictionTorqueWrench series, n=1, l_off=0).

Adds an ordinary-least-squares trend line (weighted by 1/eta_se^2) to
visualize the monotonic decrease described in the Results text, and
reports its slope for reference.
"""

import re
import numpy as np
import openpyxl
from plot_style import new_fig, COLOR_DATA, COLOR_TREND

SRC = r"C:\Users\Fabian\OneDrive - Science Tokyo\Documents\4-Software\data\Analysis\2026-06-26_HighFrictionTorqueWrench\Regression/analysis.xlsx"
OUT = r"C:\Users\Fabian\OneDrive - Science Tokyo\Documents\5-Presentation/figures/plot_eta_torque.png"

# ─── Load data ────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
ws = wb["fitted_params_per_config"]
rows = list(ws.iter_rows(values_only=True))

data_rows = [r for r in rows[1:9]]  # 8 torque conditions

torque = np.array([float(re.search(r"([\d.]+)Nm", r[0]).group(1)) for r in data_rows])
eta = np.array([r[7] for r in data_rows], dtype=float)
eta_se = np.array([r[8] for r in data_rows], dtype=float)

order = np.argsort(torque)
torque, eta, eta_se = torque[order], eta[order], eta_se[order]

# ─── Weighted linear trend (weights = 1/se^2) ────────────────────────────
w = 1.0 / eta_se**2
X = np.vstack([torque, np.ones_like(torque)]).T
W = np.diag(w)
beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ eta)
slope, intercept = beta
torque_smooth = np.linspace(torque.min(), torque.max(), 100)
eta_trend = slope * torque_smooth + intercept

# Pearson correlation for reference
r = np.corrcoef(torque, eta)[0, 1]

# ─── Plot ─────────────────────────────────────────────────────────────────
fig, ax = new_fig()

ax.plot(torque_smooth, eta_trend, "--", color=COLOR_TREND, linewidth=1.6, zorder=2,
        label=rf"Linear trend ($r$ = {r:.2f})")

ax.errorbar(torque, eta, yerr=eta_se, fmt="o", color=COLOR_DATA,
            ecolor=COLOR_DATA, elinewidth=1.2, capsize=3.5, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            label=r"Measured $\eta$ (mean $\pm$ SE)")

ax.set_xlabel("Joint preload torque (N·m)")
ax.set_ylabel(r"Per-unit efficiency $\eta$")
ax.set_title(r"Efficiency $\eta$ vs. Joint Preload Torque")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.80, 0.98)

ax.legend(loc="upper right")

fig.savefig(OUT)
print(f"Saved -> {OUT}")
print(f"torque: {torque}")
print(f"eta: {eta}")
print(f"eta_se: {eta_se}")
print(f"slope = {slope:.4f} per N·m, intercept = {intercept:.4f}, r = {r:.3f}")
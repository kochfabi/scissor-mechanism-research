"""
make_plot_eta_theta.py
-----------------------
Regenerates plot_eta_theta.png (per-unit efficiency eta vs. opening
half-angle theta) directly from analysis_eta_theta.xlsx, using a
consistent, publication-quality style.

Source data: /mnt/user-data/uploads/analysis_eta_theta.xlsx, sheet 'analysis'
  - theta            [deg]   (col D)
  - eta data         [-]     (col J) measured eta = sqrt(epsilon), n=2 units,
                              averaged per l_offset condition
  - eta_se           [-]     (col N) standard error of the mean, per condition
  - C, l, r_inner, mu        model constants (row 8): C = 4*mu*r_inner / l

Model: eta(theta) = 1 - C / sin(theta)   [equivalent to eta(theta) = 1 - 4*mu*r_inner/(l*sin(theta))]
This is verified to reproduce the 'eta fitted' column exactly (max abs
deviation < 1e-6), confirming eta_fitted is the Coulomb-friction model
evaluated at each tested angle using mu fit to the pooled data.
"""

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from plot_style import new_fig, COLOR_DATA, COLOR_MODEL

SRC = r"C:\Users\Fabian\OneDrive - Science Tokyo\Documents\4-Software\data\Analysis\2026-06-15_OpeningAngleVariation\Regression/analysis_eta_theta.xlsx"
OUT = r"C:\Users\Fabian\OneDrive - Science Tokyo\Documents\5-Presentation/figures/plot_eta_theta.png"

# ─── Load data ────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
ws = wb["analysis"]
rows = list(ws.iter_rows(values_only=True))

data_rows = [r for r in rows[1:6]]  # the 5 configuration rows
theta = np.array([r[3] for r in data_rows], dtype=float)
eta_fitted = np.array([r[4] for r in data_rows], dtype=float)
eta_data = np.array([r[9] for r in data_rows], dtype=float)
eta_se = np.array([r[13] for r in data_rows], dtype=float)

C, l_bar, r_inner, mu = rows[8][0], rows[8][1], rows[8][2], rows[8][3]

# Sort by ascending theta for a clean curve
order = np.argsort(theta)
theta, eta_fitted, eta_data, eta_se = (a[order] for a in (theta, eta_fitted, eta_data, eta_se))

# Sanity check: model reproduces eta_fitted
model_check = 1 - C / np.sin(np.deg2rad(theta))
assert np.max(np.abs(model_check - eta_fitted)) < 1e-6, "Model does not reproduce eta_fitted"

# ─── Continuous theoretical model curve ──────────────────────────────────
theta_smooth = np.linspace(theta.min() - 4, theta.max() + 4, 300)
eta_smooth = 1 - C / np.sin(np.deg2rad(theta_smooth))

# ─── Plot ─────────────────────────────────────────────────────────────────
fig, ax = new_fig()

ax.plot(theta_smooth, eta_smooth, "-", color=COLOR_MODEL, linewidth=1.8, zorder=2,
        label=r"Fitted $\eta(\theta)$")

ax.errorbar(theta, eta_data, yerr=eta_se, fmt="o", color=COLOR_DATA,
            ecolor=COLOR_DATA, elinewidth=1.2, capsize=3.5, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3,
            label=r"Measured $\eta$ (mean $\pm$ SE)")

ax.set_xlabel(r"Opening half-angle $\theta$ ($^\circ$)")
ax.set_ylabel(r"Per-unit efficiency $\eta$")
ax.set_title(r"Efficiency $\eta$ vs. Opening Angle $\theta$")
ax.set_xlim(theta_smooth.min(), theta_smooth.max())
ax.set_ylim(0.94, 1.00)

ax.legend(loc="lower right")

fig.savefig(OUT)
print(f"Saved -> {OUT}")
print(f"theta: {theta}")
print(f"eta_data: {eta_data}")
print(f"eta_se: {eta_se}")
print(f"eta_fitted (model check): {eta_fitted}")
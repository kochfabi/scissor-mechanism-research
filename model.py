import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch

# ─── User Inputs ────────────────────────────────────────────────────────────────
# Initialize model parameters
eta_init = 0.83
F_guide_init = 13.8
F_joint_init = 0
F_tare_init = 7.5
n_init = 2
l_offset_init = 10.0

# Generate ranges for a smooth theoretical layout
F_in_space = np.linspace(50, 500, 200)
n_space = np.arange(1, 5, 1)
l_off_space = np.linspace(0, 24, 25)
F_tare_space = np.linspace(0, 30, 5)

# Path to the metrics CSV file used for overlaying experimental data points.
# Change this file path to the desired data file.
#metrics_file = "data/Analysis/2026-06-15_OpeningAngleVariation/metrics.csv"
metrics_file = "data/Analysis/2026-06-16_HighFrictionTest/metrics.csv"
if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
else:
    metrics_df = pd.DataFrame()

# ─── Initialize figure layout ────────────────────────────────────────────────────────────────
fig_main = plt.figure(figsize=(12, 11))

# Main gridspec for overall layout
gs_main = fig_main.add_gridspec(2, 1, height_ratios=[3, 0.4], hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.05)

# Gridspec for 2x2 plots with normal spacing
gs_plots = gs_main[0].subgridspec(2, 2, hspace=0.5, wspace=0.25)

# Gridspec for sliders with tight spacing
gs_sliders = gs_main[1].subgridspec(5, 1, hspace=0.02, wspace=0.25)

# Create 2x2 plot axes
axs_main = np.empty((2, 2), dtype=object)
axs_main[0, 0] = fig_main.add_subplot(gs_plots[0, 0])
axs_main[0, 1] = fig_main.add_subplot(gs_plots[0, 1])
axs_main[1, 0] = fig_main.add_subplot(gs_plots[1, 0])
axs_main[1, 1] = fig_main.add_subplot(gs_plots[1, 1])

fig_side, axs_side = plt.subplots(1, 3, figsize=(18, 5))

# ─── Calculate F_out based on theoretical model ────────────────────────────────────────────────────────────────
def calc_F_out(F_in, eta, F_guide, F_joint, F_tare, n, is_loading=True):
    sign = -1.0 if is_loading else 1.0
    F_static = F_guide + n * F_joint
    return (eta**n) * F_in + sign * F_static + F_tare

# ─── Create plots to update dynamically later ────────────────────────────────────────────────────────────────
# Plot 1: F_out/F_in vs F_in
F_out_load1 = calc_F_out(F_in_space, eta_init, F_guide_init, F_joint_init, F_tare_init, n_init, True)
F_out_unload1 = calc_F_out(F_in_space, eta_init, F_guide_init, F_joint_init, F_tare_init, n_init, False)
line1_load, = axs_main[0, 0].plot(F_in_space, F_out_load1/F_in_space, 'b--')
line1_unload, = axs_main[0, 0].plot(F_in_space, F_out_unload1/F_in_space, 'r--')
axs_main[0, 0].set_title(r"Efficiency $\epsilon$ vs $F_{in}$")
axs_main[0, 0].set_xlabel("$F_{in}$ (g)")
axs_main[0, 0].set_ylabel(r"Efficiency $\epsilon = F_{out}/F_{in}$")
axs_main[0, 0].grid(True, linestyle=':', alpha=0.6)
line1_ideal = axs_main[0, 0].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 2: F_out - F_in vs F_in
line2_load, = axs_main[0, 1].plot(F_in_space, F_in_space - F_out_load1, 'b--')
line2_unload, = axs_main[0, 1].plot(F_in_space, F_in_space - F_out_unload1, 'r--')
axs_main[0, 1].set_title(r"Force Loss $F_{loss}$ vs $F_{in}$")
axs_main[0, 1].set_xlabel("$F_{in}$ (g)")
axs_main[0, 1].set_ylabel(r"Force Loss $F_{loss} = F_{in} - F_{out}$ (g)")
axs_main[0, 1].yaxis.set_inverted(True)
axs_main[0, 1].grid(True, linestyle=':', alpha=0.6)
line2_ideal = axs_main[0, 1].axhline(0.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# ────────────────────────────────────────────────────────────────
# Arrow indicators for direction on hysteresis plots
# choose stable x positions (15% and 85% of F_in range) and a horizontal offset for arrow length
_arrow_x_right = F_in_space[int(len(F_in_space) * 0.85)]
_arrow_x_left = F_in_space[int(len(F_in_space) * 0.15)]
# horizontal arrow helper (legacy values retained)
_arrow_dx = (F_in_space[1] - F_in_space[0]) * 40  # horizontal arrow length in data units

# center x and small arrow half-length
_center_idx = len(F_in_space) // 2
_x_center = F_in_space[_center_idx]
_arrow_half_len = (F_in_space.max() - F_in_space.min()) * 0.02

def _make_center_tangent_arrow(ax, x_center, y_vals, x_vals, color, reverse=False):
    y_center = np.interp(x_center, x_vals, y_vals)
    slope = np.interp(x_center, x_vals, np.gradient(y_vals, x_vals))
    theta = np.arctan(slope)
    dx = _arrow_half_len * np.cos(theta)
    dy = _arrow_half_len * np.sin(theta)
    start = (x_center - dx, y_center - dy)
    end = (x_center + dx, y_center + dy)
    if reverse:
        start, end = end, start
    # filled, slightly larger tip
    arr = FancyArrowPatch(start, end,
                         arrowstyle='Simple,head_length=10,head_width=8,tail_width=1',
                         edgecolor=color, facecolor=color,
                         mutation_scale=0.7, linewidth=0.6, zorder=5)
    ax.add_patch(arr)
    return arr

eps_load = F_out_load1 / F_in_space
eps_unload = F_out_unload1 / F_in_space
arrow1_load = _make_center_tangent_arrow(axs_main[0, 0], _x_center, eps_load, F_in_space, 'b', reverse=False)
arrow1_unload = _make_center_tangent_arrow(axs_main[0, 0], _x_center, eps_unload, F_in_space, 'r', reverse=True)

# initial y positions for plot 2
_y2_load_right = np.interp(_arrow_x_right, F_in_space, F_in_space - F_out_load1)
_y2_unload_left = np.interp(_arrow_x_left, F_in_space, F_in_space - F_out_unload1)
loss_load = F_in_space - F_out_load1
loss_unload = F_in_space - F_out_unload1
arrow2_load = _make_center_tangent_arrow(axs_main[0, 1], _x_center, loss_load, F_in_space, 'b', reverse=False)
arrow2_unload = _make_center_tangent_arrow(axs_main[0, 1], _x_center, loss_unload, F_in_space, 'r', reverse=True)
# ────────────────────────────────────────────────────────────────

# Plot 3: F_out/F_in vs l_offset (proxy for theta) in separate figure
# Assuming linear increase of F_joint with increasing l_offset based on notes
def get_F_joint_theta(f_j, l_off):
    return f_j * (1.0 + l_off / 24.0)

f_j_theta = get_F_joint_theta(F_joint_init, l_off_space)
F_out_load3 = calc_F_out(300, eta_init, F_guide_init, f_j_theta, F_tare_init, n_init, True)
line3_load, = axs_side[0].plot(l_off_space, F_out_load3/300, 'b--')
axs_side[0].set_title(r"Efficiency $\epsilon$ vs $l_{offset}$ [at $F_{in}=300g$]")
axs_side[0].set_xlabel("$l_{offset}$ (mm)")
axs_side[0].set_ylabel(r"Efficiency $\epsilon = F_{out}/F_{in}$")
axs_side[0].grid(True, linestyle=':', alpha=0.6)
line3_ideal = axs_side[0].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 4: F_out/F_in vs n in separate figure
F_out_load4 = calc_F_out(300, eta_init, F_guide_init, F_joint_init, F_tare_init, n_space, True)
line4_load, = axs_side[1].plot(n_space, F_out_load4/300, 'b--')
axs_side[1].set_title(r"Efficiency $\epsilon$ vs $n$ [at $F_{in}=300g$]")
axs_side[1].set_xlabel("Number of Links $n$")
axs_side[1].set_ylabel(r"Efficiency $\epsilon = F_{out}/F_{in}$")
axs_side[1].set_xticks(n_space)
axs_side[1].grid(True, linestyle=':', alpha=0.6)
line4_ideal = axs_side[1].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 7: average efficiency (epsilon_load+epsilon_unload)/2 vs F_in
F_out_load7 = calc_F_out(F_in_space, eta_init, F_guide_init, F_joint_init, F_tare_init, n_init, True)
F_out_unload7 = calc_F_out(F_in_space, eta_init, F_guide_init, F_joint_init, F_tare_init, n_init, False)
eps_avg_theory = 0.5 * ((F_out_load7 / F_in_space) + (F_out_unload7 / F_in_space))
line7_avg, = axs_side[2].plot(F_in_space, eps_avg_theory, 'g--', label='Model average')
axs_side[2].set_title(r"Average Efficiency $\bar{\epsilon}$ vs $F_{in}$")
axs_side[2].set_xlabel('$F_{in}$ (g)')
axs_side[2].set_ylabel(r"Average Efficiency $\bar{\epsilon}=\frac{\epsilon_{load}+\epsilon_{unload}}{2}$")
axs_side[2].grid(True, linestyle=':', alpha=0.6)
line7_ideal = axs_side[2].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label=r'Ideal $\bar{\epsilon}=1$')

# Plot 5: hysteresis absolute vs F_in
hyst_abs_init = 2.0 * (F_guide_init + n_init * F_joint_init)
line5_abs, = axs_main[1, 0].plot(F_in_space, np.full_like(F_in_space, hyst_abs_init), 'm--', label='Model ΔF')
axs_main[1, 0].set_title(r'Hysteresis Absolute ΔF vs $F_{in}$')
axs_main[1, 0].set_xlabel('$F_{in}$ (g)')
axs_main[1, 0].set_ylabel(r'Hysteresis $\Delta F = F_{out}^{(unload)} - F_{out}^{(load)}$ (g)')
axs_main[1, 0].set_ylim(0, hyst_abs_init * 2)
axs_main[1, 0].grid(True, linestyle=':', alpha=0.6)

# Plot 6: hysteresis normalized vs F_in
line6_norm, = axs_main[1, 1].plot(F_in_space, np.full_like(F_in_space, hyst_abs_init) / F_in_space, 'm--', label='Model H_norm')
axs_main[1, 1].set_title(r'Hysteresis Normalized H vs $F_{in}$')
axs_main[1, 1].set_xlabel('$F_{in}$ (g)')
axs_main[1, 1].set_ylabel(r'Normalized Hysteresis $H= ΔF/F_{in}$')
axs_main[1, 1].set_ylim(0, 0.4)
axs_main[1, 1].grid(True, linestyle=':', alpha=0.6)

# ─── Overlay metric.csv data points on plots ────────────────────────────────────────────────────────────────
if not metrics_df.empty:
    metrics_df.columns = metrics_df.columns.str.strip()
    metrics_df = metrics_df[metrics_df['F_in_g'].notna()]
    load_points = metrics_df.dropna(subset=['F_in_g', 'epsilon_load', 'F_out_load_g'])
    unload_points = metrics_df.dropna(subset=['F_in_g', 'epsilon_unload', 'F_out_unload_g'])
    points_at_300 = load_points[load_points['F_in_g'] == 300]
    avg_points_at_300 = points_at_300.groupby('n_units', as_index=False)['epsilon_load'].mean()

    scatter1_load = axs_main[0, 0].scatter(load_points['F_in_g'], load_points['epsilon_load'], facecolors='none', edgecolors='b', marker='o', s=30, label='Data load')
    scatter1_unload = axs_main[0, 0].scatter(unload_points['F_in_g'], unload_points['epsilon_unload'], facecolors='none', edgecolors='r', marker='s', s=30, label='Data unload')

    scatter2_load = axs_main[0, 1].scatter(load_points['F_in_g'], load_points['F_in_g'] - load_points['F_out_load_g'], facecolors='none', edgecolors='b', marker='o', s=30)
    scatter2_unload = axs_main[0, 1].scatter(unload_points['F_in_g'], unload_points['F_in_g'] - unload_points['F_out_unload_g'], facecolors='none', edgecolors='r', marker='s', s=30)

    scatter3_load = axs_side[0].scatter(points_at_300['l_offset'], points_at_300['epsilon_load'], facecolors='none', edgecolors='b', marker='o', s=20, label='Data load $F_{in}=300$')
    scatter4_load = axs_side[1].scatter(avg_points_at_300['n_units'], avg_points_at_300['epsilon_load'], facecolors='none', edgecolors='k', marker='s', s=20, label='Data avg $F_{in}=300$')
    avg_points = metrics_df.dropna(subset=['F_in_g', 'epsilon_load', 'epsilon_unload'])
    if not avg_points.empty:
        scatter7_avg = axs_side[2].scatter(
            avg_points['F_in_g'],
            0.5 * (avg_points['epsilon_load'] + avg_points['epsilon_unload']),
            facecolors='none', edgecolors='g', marker='o', s=30, label='Data avg')
    scatter5 = axs_main[1, 0].scatter(load_points['F_in_g'], load_points['delta_F_g'], facecolors='none', edgecolors='m', marker='D', s=20, label='Data ΔF')
    scatter6 = axs_main[1, 1].scatter(load_points['F_in_g'], load_points['H_pct'], facecolors='none', edgecolors='m', marker='D', s=20, label='Data H_pct')

# ─── Interactive Sliders Configuration ────────────────────────────────────────────────────────────────
ax_eta = fig_main.add_subplot(gs_sliders[0])
ax_F_guide = fig_main.add_subplot(gs_sliders[1])
ax_F_joint = fig_main.add_subplot(gs_sliders[2])
ax_F_tare = fig_main.add_subplot(gs_sliders[3])
ax_n = fig_main.add_subplot(gs_sliders[4])

slider_eta = Slider(ax_eta, r'Efficiency $\eta$', 0.70, 1.0, valinit=eta_init, valfmt='%1.2f')
slider_F_guide = Slider(ax_F_guide, '$F_{guide}$ (g)', 0.0, 50.0, valinit=F_guide_init, valfmt='%1.1f')
slider_F_joint = Slider(ax_F_joint, '$F_{joint}$ (g)', 0.0, 20.0, valinit=F_joint_init, valfmt='%1.1f')
slider_F_tare = Slider(ax_F_tare, '$F_{tare}$ (g)', 0.0, 30.0, valinit=F_tare_init, valfmt='%1.1f')
slider_n = Slider(ax_n, '$n$', 1, 4, valinit=n_init, valstep=1, valfmt='%0.0f')

def update_plots(val):
    eta = slider_eta.val
    F_guide = slider_F_guide.val
    F_joint = slider_F_joint.val
    F_tare = slider_F_tare.val
    n_current = int(slider_n.val)
    
    # Update Plot 1 & 2 in the main interactive figure
    l_load = calc_F_out(F_in_space, eta, F_guide, F_joint, F_tare, n_current, True)
    ul_load = calc_F_out(F_in_space, eta, F_guide, F_joint, F_tare, n_current, False)
    
    line1_load.set_ydata(l_load / F_in_space)
    line1_unload.set_ydata(ul_load / F_in_space)
    line2_load.set_ydata(F_in_space - l_load)
    line2_unload.set_ydata(F_in_space - ul_load)

    # ────────────────────────────────────────────────────────────────
    # Update arrow positions for plots 1 & 2
    try:
        # plot 1 (efficiency)
        eps_load_up = l_load / F_in_space
        eps_unload_up = ul_load / F_in_space
        slope_load = np.interp(_x_center, F_in_space, np.gradient(eps_load_up, F_in_space))
        slope_unload = np.interp(_x_center, F_in_space, np.gradient(eps_unload_up, F_in_space))
        theta_load = np.arctan(slope_load)
        theta_unload = np.arctan(slope_unload)
        dx_load = _arrow_half_len * np.cos(theta_load)
        dy_load = _arrow_half_len * np.sin(theta_load)
        dx_un = _arrow_half_len * np.cos(theta_unload)
        dy_un = _arrow_half_len * np.sin(theta_unload)
        y_center_load = np.interp(_x_center, F_in_space, eps_load_up)
        y_center_un = np.interp(_x_center, F_in_space, eps_unload_up)
        arrow1_load.set_positions((_x_center - dx_load, y_center_load - dy_load), (_x_center + dx_load, y_center_load + dy_load))
        # unloading arrow should point left: set positions reversed
        arrow1_unload.set_positions((_x_center + dx_un, y_center_un + dy_un), (_x_center - dx_un, y_center_un - dy_un))

        # plot 2 (loss)
        loss_load_up = F_in_space - l_load
        loss_unload_up = F_in_space - ul_load
        slope_l_load = np.interp(_x_center, F_in_space, np.gradient(loss_load_up, F_in_space))
        slope_l_un = np.interp(_x_center, F_in_space, np.gradient(loss_unload_up, F_in_space))
        th_l_load = np.arctan(slope_l_load)
        th_l_un = np.arctan(slope_l_un)
        dx_l = _arrow_half_len * np.cos(th_l_load)
        dy_l = _arrow_half_len * np.sin(th_l_load)
        dx_l_un = _arrow_half_len * np.cos(th_l_un)
        dy_l_un = _arrow_half_len * np.sin(th_l_un)
        y_c_l = np.interp(_x_center, F_in_space, loss_load_up)
        y_c_l_un = np.interp(_x_center, F_in_space, loss_unload_up)
        arrow2_load.set_positions((_x_center - dx_l, y_c_l - dy_l), (_x_center + dx_l, y_c_l + dy_l))
        # unloading arrow should point left: set positions reversed
        arrow2_unload.set_positions((_x_center + dx_l_un, y_c_l_un + dy_l_un), (_x_center - dx_l_un, y_c_l_un - dy_l_un))
    except Exception:
        pass
    # ────────────────────────────────────────────────────────────────

    # Update the separate static plots 3 and 4 for results
    fj_t = get_F_joint_theta(F_joint, l_off_space)
    l_load3 = calc_F_out(300, eta, F_guide, fj_t, F_tare, n_current, True)
    line3_load.set_ydata(l_load3 / 300)

    l_load4 = calc_F_out(300, eta, F_guide, F_joint, F_tare, n_space, True)
    line4_load.set_ydata(l_load4 / 300)

    # Update Plot 7: average efficiency (epsilon_load+epsilon_unload)/2 vs F_in
    l_load7 = calc_F_out(F_in_space, eta, F_guide, F_joint, F_tare, n_current, True)
    ul_load7 = calc_F_out(F_in_space, eta, F_guide, F_joint, F_tare, n_current, False)
    eps_avg_up = 0.5 * ((l_load7 / F_in_space) + (ul_load7 / F_in_space))
    line7_avg.set_ydata(eps_avg_up)

    # Update hysteresis plots
    hyst_abs = 2.0 * (F_guide + n_current * F_joint)
    line5_abs.set_ydata(np.full_like(F_in_space, hyst_abs))
    line6_norm.set_ydata(np.full_like(F_in_space, hyst_abs) / F_in_space)
    
    # Redraw frame canvas without rescaling existing axes limits
    fig_main.canvas.draw_idle()
    fig_side.canvas.draw_idle()

slider_eta.on_changed(update_plots)
slider_F_guide.on_changed(update_plots)
slider_F_joint.on_changed(update_plots)
slider_F_tare.on_changed(update_plots)
slider_n.on_changed(update_plots)

axs_main[0, 0].legend(loc='lower right')
axs_main[0, 1].legend(loc='upper right')
axs_main[1, 0].legend(loc='upper right')
axs_main[1, 1].legend(loc='upper right')
axs_side[0].legend(loc='upper right')
axs_side[1].legend(loc='upper right')
axs_side[2].legend(loc='upper right')
plt.show()
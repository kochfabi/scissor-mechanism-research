import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initialize baseline parameters
eta_init = 0.99
f_guide_init = 8.5
f_joint_init = 1.5
n_init = 2
l_offset_init = 10.0  # representing geometry proxy for theta

# Generate dummy ranges for a smooth theoretical layout
F_in_space = np.linspace(50, 500, 200)
n_space = np.arange(1, 5, 1)
l_off_space = np.linspace(0, 24, 25)

# Path to the metrics CSV file used for overlaying experimental data points.
# Change this file path to the desired data file.
metrics_file = "data/Analysis/2026-06-16_HighFrictionTest/metrics.csv"
if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
else:
    metrics_df = pd.DataFrame()

# Initialize figure layout with separate gridspecs for plots and sliders
fig_main = plt.figure(figsize=(12, 11))

# Main gridspec for overall layout
gs_main = fig_main.add_gridspec(2, 1, height_ratios=[3, 0.4], hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.05)

# Gridspec for 2x2 plots with normal spacing
gs_plots = gs_main[0].subgridspec(2, 2, hspace=0.5, wspace=0.25)

# Gridspec for sliders with tight spacing
gs_sliders = gs_main[1].subgridspec(4, 1, hspace=0.02, wspace=0.25)

# Create 2x2 plot axes
axs_main = np.empty((2, 2), dtype=object)
axs_main[0, 0] = fig_main.add_subplot(gs_plots[0, 0])
axs_main[0, 1] = fig_main.add_subplot(gs_plots[0, 1])
axs_main[1, 0] = fig_main.add_subplot(gs_plots[1, 0])
axs_main[1, 1] = fig_main.add_subplot(gs_plots[1, 1])

fig_side, axs_side = plt.subplots(1, 2, figsize=(12, 5))
fig_side.suptitle('Separate Plots: 3 and 4', fontsize=14)

# --- Define equations based on your theoretical model ---
def calc_F_out(F_in, eta, f_guide, f_joint, n, is_loading=True):
    sign = -1.0 if is_loading else 1.0
    F_static = f_guide + n * f_joint
    return (eta**n) * F_in + sign * F_static

# Create plot objects to update dynamically later
# Plot 1: F_out/F_in vs F_in
F_out_load1 = calc_F_out(F_in_space, eta_init, f_guide_init, f_joint_init, n_init, True)
F_out_unload1 = calc_F_out(F_in_space, eta_init, f_guide_init, f_joint_init, n_init, False)
line1_load, = axs_main[0, 0].plot(F_in_space, F_out_load1/F_in_space, 'b-', label='Loading')
line1_unload, = axs_main[0, 0].plot(F_in_space, F_out_unload1/F_in_space, 'r-', label='Unloading')
axs_main[0, 0].set_title(r"$\epsilon$ ($F_{out}/F_{in}$) vs $F_{in}$")
axs_main[0, 0].set_xlabel("$F_{in}$ (g)")
axs_main[0, 0].set_ylabel(r"Efficiency $\epsilon$")
axs_main[0, 0].grid(True, linestyle=':', alpha=0.6)
line1_ideal = axs_main[0, 0].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 2: F_out - F_in vs F_in
line2_load, = axs_main[0, 1].plot(F_in_space, F_out_load1 - F_in_space, 'b-')
line2_unload, = axs_main[0, 1].plot(F_in_space, F_out_unload1 - F_in_space, 'r-')
axs_main[0, 1].set_title(r"$F_{out} - F_{in}$ vs $F_{in}$")
axs_main[0, 1].set_xlabel("$F_{in}$ (g)")
axs_main[0, 1].set_ylabel(r"Force Difference $\Delta F$ (g)")
axs_main[0, 1].grid(True, linestyle=':', alpha=0.6)
line2_ideal = axs_main[0, 1].axhline(0.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 3: F_out/F_in vs l_offset (proxy for theta) in separate figure
# Assuming linear increase of f_joint with increasing l_offset based on notes
def get_f_joint_theta(f_j, l_off):
    return f_j * (1.0 + l_off / 24.0)

f_j_theta = get_f_joint_theta(f_joint_init, l_off_space)
F_out_load3 = calc_F_out(300, eta_init, f_guide_init, f_j_theta, n_init, True)
line3_load, = axs_side[0].plot(l_off_space, F_out_load3/300, 'b-')
axs_side[0].set_title(r"$\epsilon$ vs $l_{offset}$ ($\theta$) [at $F_{in}=300g$]")
axs_side[0].set_xlabel("$l_{offset}$ (mm)")
axs_side[0].set_ylabel(r"Efficiency $\epsilon$")
axs_side[0].grid(True, linestyle=':', alpha=0.6)
line3_ideal = axs_side[0].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 4: F_out/F_in vs n in separate figure
F_out_load4 = calc_F_out(300, eta_init, f_guide_init, f_joint_init, n_space, True)
line4_load, = axs_side[1].plot(n_space, F_out_load4/300, 'b-o')
axs_side[1].set_title(r"$\epsilon$ vs $n$ [at $F_{in}=300g$]")
axs_side[1].set_xlabel("Number of Links ($n$)")
axs_side[1].set_ylabel(r"Efficiency $\epsilon$")
axs_side[1].set_xticks(n_space)
axs_side[1].grid(True, linestyle=':', alpha=0.6)
line4_ideal = axs_side[1].axhline(1.0, color='k', linestyle=':', linewidth=1.5, label='Ideal $F_{out}=F_{in}$')

# Plot 5: hysteresis absolute vs F_in
hyst_abs_init = 2.0 * (f_guide_init + n_init * f_joint_init)
line5_abs, = axs_main[1, 0].plot(F_in_space, np.full_like(F_in_space, hyst_abs_init), 'm-', label='Model ΔF')
axs_main[1, 0].set_title(r'Hysteresis Absolute ΔF vs $F_{in}$')
axs_main[1, 0].set_xlabel('$F_{in}$ (g)')
axs_main[1, 0].set_ylabel('Hysteresis ΔF (g)')
axs_main[1, 0].set_ylim(0, hyst_abs_init * 2)
axs_main[1, 0].grid(True, linestyle=':', alpha=0.6)

# Plot 6: hysteresis normalized vs F_in
line6_norm, = axs_main[1, 1].plot(F_in_space, np.full_like(F_in_space, hyst_abs_init) / F_in_space, 'm-', label='Model H_norm')
axs_main[1, 1].set_title(r'Hysteresis Normalized $ΔF/F_{in}$ vs $F_{in}$')
axs_main[1, 1].set_xlabel('$F_{in}$ (g)')
axs_main[1, 1].set_ylabel('Normalized Hysteresis')
axs_main[1, 1].set_ylim(0, 0.4)
axs_main[1, 1].grid(True, linestyle=':', alpha=0.6)

# Overlay specific metric.csv data points on the plots.
if not metrics_df.empty:
    metrics_df.columns = metrics_df.columns.str.strip()
    metrics_df = metrics_df[metrics_df['F_in_g'].notna()]
    load_points = metrics_df.dropna(subset=['F_in_g', 'epsilon_load', 'F_out_load_g'])
    unload_points = metrics_df.dropna(subset=['F_in_g', 'epsilon_unload', 'F_out_unload_g'])
    points_at_300 = load_points[load_points['F_in_g'] == 300]
    avg_points_at_300 = points_at_300.groupby('n_units', as_index=False)['epsilon_load'].mean()

    scatter1_load = axs_main[0, 0].scatter(load_points['F_in_g'], load_points['epsilon_load'], facecolors='none', edgecolors='b', marker='o', s=60, label='Data load')
    scatter1_unload = axs_main[0, 0].scatter(unload_points['F_in_g'], unload_points['epsilon_unload'], facecolors='none', edgecolors='r', marker='^', s=60, label='Data unload')

    scatter2_load = axs_main[0, 1].scatter(load_points['F_in_g'], load_points['F_out_load_g'] - load_points['F_in_g'], facecolors='none', edgecolors='b', marker='o', s=60)
    scatter2_unload = axs_main[0, 1].scatter(unload_points['F_in_g'], unload_points['F_out_unload_g'] - unload_points['F_in_g'], facecolors='none', edgecolors='r', marker='^', s=60)

    scatter3_load = axs_side[0].scatter(points_at_300['l_offset'], points_at_300['epsilon_load'], facecolors='none', edgecolors='b', marker='o', s=70, label='Data load $F_{in}=300$')
    scatter4_load = axs_side[1].scatter(avg_points_at_300['n_units'], avg_points_at_300['epsilon_load'], facecolors='none', edgecolors='k', marker='s', s=80, label='Data avg $F_{in}=300$')
    scatter5 = axs_main[1, 0].scatter(load_points['F_in_g'], load_points['delta_F_g'], facecolors='none', edgecolors='m', marker='D', s=70, label='Data ΔF')
    scatter6 = axs_main[1, 1].scatter(load_points['F_in_g'], load_points['H_pct'], facecolors='none', edgecolors='m', marker='D', s=70, label='Data H_pct')

# --- Interactive Sliders Configuration ---
ax_eta = fig_main.add_subplot(gs_sliders[0])
ax_f_guide = fig_main.add_subplot(gs_sliders[1])
ax_f_joint = fig_main.add_subplot(gs_sliders[2])
ax_n = fig_main.add_subplot(gs_sliders[3])

slider_eta = Slider(ax_eta, r'Efficiency $\eta$', 0.70, 1.0, valinit=eta_init, valfmt='%1.2f')
slider_f_guide = Slider(ax_f_guide, '$F_{guide}$ (g)', 0.0, 50.0, valinit=f_guide_init, valfmt='%1.1f')
slider_f_joint = Slider(ax_f_joint, '$f_{joint}$ (g)', 0.0, 20.0, valinit=f_joint_init, valfmt='%1.1f')
slider_n = Slider(ax_n, '$n$', 1, 4, valinit=n_init, valstep=1, valfmt='%0.0f')

def update_plots(val):
    eta = slider_eta.val
    f_guide = slider_f_guide.val
    f_joint = slider_f_joint.val
    n_current = int(slider_n.val)
    
    # Update Plot 1 & 2 in the main interactive figure
    l_load = calc_F_out(F_in_space, eta, f_guide, f_joint, n_current, True)
    ul_load = calc_F_out(F_in_space, eta, f_guide, f_joint, n_current, False)
    
    line1_load.set_ydata(l_load / F_in_space)
    line1_unload.set_ydata(ul_load / F_in_space)
    line2_load.set_ydata(l_load - F_in_space)
    line2_unload.set_ydata(ul_load - F_in_space)

    # Update the separate static plots 3 and 4 for results
    fj_t = get_f_joint_theta(f_joint, l_off_space)
    l_load3 = calc_F_out(300, eta, f_guide, fj_t, n_current, True)
    line3_load.set_ydata(l_load3 / 300)

    l_load4 = calc_F_out(300, eta, f_guide, f_joint, n_space, True)
    line4_load.set_ydata(l_load4 / 300)

    # Update hysteresis plots
    hyst_abs = 2.0 * (f_guide + n_current * f_joint)
    line5_abs.set_ydata(np.full_like(F_in_space, hyst_abs))
    line6_norm.set_ydata(np.full_like(F_in_space, hyst_abs) / F_in_space)
    
    # Redraw frame canvas cleanly
    for ax in axs_main.ravel():
        ax.relim()
        ax.autoscale_view()
    for ax in axs_side:
        ax.relim()
        ax.autoscale_view()
    fig_main.canvas.draw_idle()
    fig_side.canvas.draw_idle()

slider_eta.on_changed(update_plots)
slider_f_guide.on_changed(update_plots)
slider_f_joint.on_changed(update_plots)
slider_n.on_changed(update_plots)

axs_main[0, 0].legend(loc='lower right')
axs_main[0, 1].legend(loc='upper right')
axs_main[1, 0].legend(loc='upper right')
axs_main[1, 1].legend(loc='upper right')
axs_side[0].legend(loc='upper right')
axs_side[1].legend(loc='upper right')
plt.show()
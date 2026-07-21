import numpy as np
import matplotlib.pyplot as plt

# Parameters
l = 0.1
l_curv = 0.005
mu = 0.082
r_inner = 0.0055
theta = np.deg2rad(50)
F_in = 100
n_stages = 1

# Original model (ignoring the asymmetric lever arm effect of l_curv)
def compute_eta_orig(mu, r_inner, l, theta):
    return 1 - (4 * mu * r_inner) / (l * np.sin(theta))

# Curvilinear model
def compute_eta_curv(mu, r_inner, l, l_curv, theta):
    numerator = l**2 - 4 * l_curv**2 - (4 * mu * l * r_inner) / np.sin(theta)
    denominator = l**2 + 4 * l_curv**2
    return numerator / denominator

fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))


# Plot 1: Transmission Efficiency vs curvature offset
l_curv_range = np.linspace(0, 0.04, 100) # up to 40mm
eta_curv_range = compute_eta_curv(mu, r_inner, l, l_curv_range, theta)

ax1.plot(l_curv_range * 1000, eta_curv_range, color='red', label='Curvilinear Model $\\eta$')
ax1.axhline(compute_eta_orig(mu, r_inner, l, theta), color='blue', linestyle='--', label='Original Model $\\eta$')
ax1.set_xlabel('Curvature Offset $l_{curv}$ (mm)')
ax1.set_ylabel('Transmission Efficiency $\\eta$')
ax1.set_title('$\\eta$ vs Curvature Offset $l_{curv}$ ($\\mu$ = 0.082, $r_{inner}$ = 5.5 mm, $\\theta$ = 50°)')
ax1.legend()
ax1.grid(True)

# Plot 2: Eta vs Theta for different curvature offsets
theta_range = np.deg2rad(np.linspace(1, 90, 100))

fig, ax2 = plt.subplots(1, 1, figsize=(7, 6))
ax2.plot(np.rad2deg(theta_range), compute_eta_curv(mu, r_inner, l, l_curv, theta_range), color='green', label=f'Curvilinear Model $\\eta$ ($l_{{curv}}$ = {l_curv*1000:.1f} mm)')
ax2.plot(np.rad2deg(theta_range), compute_eta_orig(mu, r_inner, l, theta_range), color='blue', linestyle='--', label='Original Model $\\eta$')
ax2.set_xlabel('Angle $\\theta$ ($\\degree$)')
ax2.set_ylabel('Transmission Efficiency $\\eta$')
ax2.set_ylim(0.7, 1)
ax2.set_title('$\\eta$ vs Angle $\\theta$ ($\\mu$ = 0.082, $r_{inner}$ = 5.5 mm, $l_{curv}$ = 5.0 mm)')
ax2.legend()
ax2.grid(True)


plt.tight_layout()
plt.show()
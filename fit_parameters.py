"""
fit_parameters.py
-----------------
Fits friction model parameters (eta, F_guide, F_tare) per configuration
found in metrics.csv files.

Model (F_joint = 0, single-unit guide friction only):
    F_out = eta^n * F_in  +  sign * F_guide  +  F_tare
    sign = -1 (loading), +1 (unloading)

Two-stage sequential fit per configuration:
    Stage 1 — F_guide = mean(delta_F / 2)           [closed-form, no regression]
    Stage 2 — fit eta, F_tare to eps_avg = eta^n + F_tare / F_in

A configuration is one unique (label, n_units, l_offset, l_curve) group
within the loaded metrics data.

Usage:
    python fit_parameters.py [path/to/analysis/folder]

Output:
    data/Regression/fitted_params_per_config.csv
    data/Regression/fit_plot.png
"""

import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent
DATA_DIR      = PROJECT_ROOT / "data"
ANALYSIS_DIR  = DATA_DIR / "Analysis" / "2026-05-29_DesignValidationTest"
OUTPUT_DIR    = ANALYSIS_DIR / "Regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Override analysis directory from command line if given
if len(sys.argv) > 1:
    ANALYSIS_DIR = Path(sys.argv[1])


# ─── Load metrics.csv files ───────────────────────────────────────────────────
# Each file may contain two sections; only the per-condition block is read
# (lines before "# Summary per configuration").

def load_metrics(analysis_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(analysis_dir.rglob("metrics.csv")):
        lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    if "Summary per configuration" in stripped:
                        break
                    continue
                lines.append(line)
        if len(lines) < 2:
            continue
        df = pd.read_csv(StringIO("".join(lines)))
        df.columns = df.columns.str.strip()
        df["_source"] = path.parent.name
        frames.append(df)
        print(f"Loaded ({len(df)} rows): {path.relative_to(PROJECT_ROOT)}")

    if not frames:
        raise FileNotFoundError(f"No metrics.csv files found under {analysis_dir}")

    df = pd.concat(frames, ignore_index=True)
    numeric = [
        "n_units", "l_offset", "l_curve", "F_in_g",
        "F_out_load_g", "epsilon_load",
        "F_out_unload_g", "epsilon_unload",
        "delta_F_g",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


print(f"\nReading data from: {ANALYSIS_DIR}")
df = load_metrics(ANALYSIS_DIR)
print(f"Total rows: {len(df)}\n")


# ─── Fit per configuration ────────────────────────────────────────────────────
# A configuration = unique (label, n_units, l_offset, l_curve) group.
# For each configuration:
#   Stage 1: F_guide = mean(delta_F / 2)
#   Stage 2: nonlinear fit of eps_avg = eta^n + F_tare / F_in

GROUP_KEYS = ["label", "n_units", "l_offset", "l_curve"]
results = []

for group_vals, group in df.groupby(GROUP_KEYS, dropna=False):
    label, n_units, l_offset, l_curve = group_vals

    # ── Stage 1: F_guide ──────────────────────────────────────────────────────
    delta_rows = group.dropna(subset=["delta_F_g"])
    delta_rows = delta_rows[delta_rows["delta_F_g"] > 0]

    if delta_rows.empty:
        print(f"[SKIP] {label} — no delta_F_g data for Stage 1")
        continue

    F_guide = float(delta_rows["delta_F_g"].mean() / 2.0)
    F_guide_std = float(delta_rows["delta_F_g"].std(ddof=1) / 2.0) if len(delta_rows) > 1 else float("nan")

    # ── Stage 2: eta, F_tare ──────────────────────────────────────────────────
    # Build eps_avg row-wise: average of load and unload where both exist,
    # otherwise fall back to whichever is available.
    sub = group.dropna(subset=["F_in_g"]).copy()
    sub = sub[sub["F_in_g"] > 0]

    has_l = sub["epsilon_load"].notna()
    has_u = sub["epsilon_unload"].notna()
    sub["eps_avg"] = np.nan
    sub.loc[has_l & has_u,  "eps_avg"] = 0.5 * (sub.loc[has_l & has_u, "epsilon_load"]
                                                + sub.loc[has_l & has_u, "epsilon_unload"])
    sub.loc[has_l & ~has_u, "eps_avg"] = sub.loc[has_l & ~has_u, "epsilon_load"]
    sub.loc[has_u & ~has_l, "eps_avg"] = sub.loc[has_u & ~has_l, "epsilon_unload"]
    sub = sub.dropna(subset=["eps_avg"])

    if len(sub) < 2:
        print(f"[SKIP] {label} — not enough efficiency rows for Stage 2 (need ≥ 2)")
        continue

    F_in_arr  = sub["F_in_g"].values.astype(float)
    eps_arr   = sub["eps_avg"].values.astype(float)
    n         = float(n_units)

    def model_eps(F_in, eta, F_tare):
        return eta**n + F_tare / F_in

    try:
        params, cov = curve_fit(
            model_eps, F_in_arr, eps_arr,
            p0=[0.97, 0.0],
            bounds=([0.5, -100.0], [1.0, 100.0]),
            maxfev=10_000,
        )
        eta, F_tare = float(params[0]), float(params[1])
        se = np.sqrt(np.diag(cov).clip(0))
        eta_se, F_tare_se = float(se[0]), float(se[1])

        eps_pred = model_eps(F_in_arr, eta, F_tare)
        ss_res = np.sum((eps_arr - eps_pred) ** 2)
        ss_tot = np.sum((eps_arr - eps_arr.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    except RuntimeError:
        print(f"[WARN] {label} — curve_fit did not converge, skipping.")
        continue

    results.append({
        "label"      : label,
        "n_units"    : n_units,
        "l_offset"   : l_offset,
        "l_curve"    : l_curve,
        "F_guide"    : round(F_guide,  4),
        "F_guide_std": round(F_guide_std, 4),
        "F_joint"    : 0.0,          # placeholder — extend later with multi-n data
        "eta"        : round(eta,     6),
        "eta_se"     : round(eta_se,  6),
        "F_tare"     : round(F_tare,  4),
        "F_tare_se"  : round(F_tare_se, 4),
        "r2_stage2"  : round(r2,      5),
        "n_points"   : len(sub),
    })

    print(
        f"{label}\n"
        f"  F_guide = {F_guide:.3f} ± {F_guide_std:.3f} g\n"
        f"  eta     = {eta:.5f} ± {eta_se:.5f}\n"
        f"  F_tare  = {F_tare:+.3f} ± {F_tare_se:.3f} g\n"
        f"  R²      = {r2:.4f}  (n_points = {len(sub)})\n"
    )

if not results:
    print("No configurations could be fitted. Exiting.")
    sys.exit(1)

results_df = pd.DataFrame(results)


# ─── Save ─────────────────────────────────────────────────────────────────────

csv_path = OUTPUT_DIR / "fitted_params_per_config.csv"
results_df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"Parameters saved → {csv_path.relative_to(PROJECT_ROOT)}")


# ─── Plot ─────────────────────────────────────────────────────────────────────
# One row per fitted configuration: eta, F_guide, F_tare summary.

n_configs = len(results_df)
fig, axs = plt.subplots(1, 3, figsize=(14, max(4, n_configs * 0.6 + 2)))

y_pos = np.arange(n_configs)
short_labels = [
    (lbl.split("|")[0].strip() if "|" in lbl else lbl)[:30]
    for lbl in results_df["label"]
]

# eta
axs[0].barh(y_pos, results_df["eta"], xerr=results_df["eta_se"],
            color="steelblue", ecolor="black", capsize=4, height=0.6)
axs[0].set_xlabel(r"$\eta$ per unit  [−]")
axs[0].set_title(r"Transmission efficiency $\eta$")
axs[0].set_yticks(y_pos)
axs[0].set_yticklabels(short_labels, fontsize=8)
axs[0].axvline(1.0, color="red", linestyle="--", linewidth=1, label="ideal")
axs[0].legend(fontsize=8)

# F_guide
axs[1].barh(y_pos, results_df["F_guide"], xerr=results_df["F_guide_std"],
            color="darkorange", ecolor="black", capsize=4, height=0.6)
axs[1].set_xlabel(r"$F_{guide}$  [g]")
axs[1].set_title(r"Guide friction offset $F_{guide}$")
axs[1].set_yticks(y_pos)
axs[1].set_yticklabels(short_labels, fontsize=8)

# F_tare
axs[2].barh(y_pos, results_df["F_tare"], xerr=results_df["F_tare_se"],
            color="mediumpurple", ecolor="black", capsize=4, height=0.6)
axs[2].set_xlabel(r"$F_{tare}$  [g]")
axs[2].set_title(r"Zero offset $F_{tare}$")
axs[2].set_yticks(y_pos)
axs[2].set_yticklabels(short_labels, fontsize=8)
axs[2].axvline(0.0, color="gray", linestyle="--", linewidth=1)

for ax in axs:
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.invert_yaxis()

plt.suptitle("Fitted parameters per configuration", fontsize=12, y=1.01)
plt.tight_layout()

plot_path = OUTPUT_DIR / "fit_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved      → {plot_path.relative_to(PROJECT_ROOT)}")
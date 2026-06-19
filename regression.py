from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "Analysis"

OUTPUT_DIR = DATA_DIR / "Regression"

PLOTS_DIR = OUTPUT_DIR / "plots"
MEASURED_DIR = PLOTS_DIR / "measured_vs_model"
RESIDUAL_DIR = PLOTS_DIR / "residuals"

for directory in [
    OUTPUT_DIR,
    PLOTS_DIR,
    MEASURED_DIR,
    RESIDUAL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# Helper Functions
# =========================================================
def save_fit_summary(
    eta,
    fguide,
    fjoint,
    r2,
):
    with open(
        OUTPUT_DIR / "fit_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write("Scissor Mechanism Regression Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Model:\n")
        f.write(
            "F_out = eta^n * F_in ± (fguide + n*fjoint)\n\n"
        )

        f.write(f"eta      = {eta:.6f}\n")
        f.write(f"fguide   = {fguide:.3f} g\n")
        f.write(f"fjoint   = {fjoint:.3f} g\n")
        f.write(f"R²       = {r2:.5f}\n")

# =========================================================
# Load Data
# =========================================================

from io import StringIO


def load_metrics():

    dfs = []

    for csv_path in ANALYSIS_DIR.rglob("metrics.csv"):

        try:

            lines = []

            with open(csv_path, "r", encoding="utf-8") as f:

                started = False

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    if line.startswith("#"):

                        if "Summary per configuration" in line:
                            break

                        continue

                    if not started:

                        if line.startswith("label"):
                            started = True
                            lines.append(line)

                        continue

                    lines.append(line)

            if len(lines) < 2:
                print(f"Skipping empty file: {csv_path}")
                continue

            df = pd.read_csv(
                StringIO("\n".join(lines)),
                sep=",",
            )

            df["source"] = csv_path.parent.name
            df["source_path"] = str(csv_path)

            dfs.append(df)

            print(
                f"Loaded {csv_path.name}: "
                f"{len(df)} rows"
            )

        except Exception as e:

            print(
                f"Failed to load {csv_path}: {e}"
            )

    if not dfs:
        raise RuntimeError(
            f"No usable metrics.csv files found in {ANALYSIS_DIR}"
        )

    return pd.concat(
        dfs,
        ignore_index=True,
    )


metrics = load_metrics()

print("\nColumns found:")
print(metrics.columns.tolist())

print("\nFirst rows:")
print(metrics.head())

metrics.to_csv(
    OUTPUT_DIR / "merged_metrics.csv",
    index=False,
)

print(f"\nLoaded {len(metrics)} measurements")


# =========================================================
# Validate Columns
# =========================================================

required_columns = [
    "n_units",
    "l_offset",
    "F_in_g",
    "F_out_load_g",
]

missing = [
    col
    for col in required_columns
    if col not in metrics.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns:\n{missing}"
    )


# =========================================================
# Convert Numeric Columns
# =========================================================

numeric_cols = [
    "n_units",
    "l_offset",
    "F_in_g",
    "F_out_load_g",
    "F_out_unload_g",
]

for col in numeric_cols:

    if col in metrics.columns:

        metrics[col] = pd.to_numeric(
            metrics[col],
            errors="coerce",
        )


# =========================================================
# Build Long Format Dataset
# =========================================================

records = []

for _, row in metrics.iterrows():

    n = row["n_units"]
    l_offset = row["l_offset"]
    F_in = row["F_in_g"]

    if pd.isna(F_in):
        continue

    # Loading point

    if pd.notna(row["F_out_load_g"]):

        records.append(
            {
                "n": n,
                "l_offset": l_offset,
                "F_in": F_in,
                "F_out": row["F_out_load_g"],
                "sign": -1,
                "direction": "load",
            }
        )

    # Unloading point

    if (
        "F_out_unload_g" in row
        and pd.notna(row["F_out_unload_g"])
    ):

        records.append(
            {
                "n": n,
                "l_offset": l_offset,
                "F_in": F_in,
                "F_out": row["F_out_unload_g"],
                "sign": 1,
                "direction": "unload",
            }
        )

fit_df = pd.DataFrame(records)

fit_df = fit_df.dropna(
    subset=[
        "F_in",
        "F_out",
        "n",
    ]
)

print(
    f"\nRegression points: {len(fit_df)}"
)

print(
    f"NaNs remaining in F_out: "
    f"{fit_df['F_out'].isna().sum()}"
)


# =========================================================
# Model
# =========================================================

def model(X, eta, fguide, fjoint):

    F_in, n, sign = X

    transmission = eta ** n

    friction = (
        fguide
        + fjoint * n
    )

    return (
        transmission * F_in
        + sign * friction
    )


# =========================================================
# Prepare Fit Arrays
# =========================================================

xdata = np.vstack(
    [
        fit_df["F_in"].values.astype(float),
        fit_df["n"].values.astype(float),
        fit_df["sign"].values.astype(float),
    ]
)

ydata = fit_df["F_out"].values.astype(float)

print(
    f"NaNs in ydata: "
    f"{np.isnan(ydata).sum()}"
)

print(
    f"NaNs in xdata: "
    f"{np.isnan(xdata).sum()}"
)

if np.isnan(ydata).any():
    raise RuntimeError(
        "NaNs remain in ydata."
    )

if np.isnan(xdata).any():
    raise RuntimeError(
        "NaNs remain in xdata."
    )


# =========================================================
# Fit Parameters
# =========================================================

initial_guess = [
    0.98,  # eta
    5.0,   # fguide
    2.0,   # fjoint
]

bounds = (
    [0.5, 0.0, 0.0],
    [1.0, 200.0, 100.0],
)

params, covariance = curve_fit(
    model,
    xdata,
    ydata,
    p0=initial_guess,
    bounds=bounds,
    maxfev=20000,
)

eta, fguide, fjoint = params

print("\n===== FIT RESULTS =====")
print(f"eta      = {eta:.6f}")
print(f"fguide   = {fguide:.3f} g")
print(f"fjoint   = {fjoint:.3f} g")


# =========================================================
# Predictions
# =========================================================

fit_df["prediction"] = model(
    xdata,
    eta,
    fguide,
    fjoint,
)

fit_df["residual"] = (
    fit_df["F_out"]
    - fit_df["prediction"]
)


# =========================================================
# Goodness of Fit
# =========================================================

ss_res = np.sum(
    (fit_df["F_out"] - fit_df["prediction"]) ** 2
)

ss_tot = np.sum(
    (fit_df["F_out"] - fit_df["F_out"].mean()) ** 2
)

r2 = 1 - ss_res / ss_tot

print(f"R² = {r2:.5f}")


# =========================================================
# Save Parameters
# =========================================================

results = pd.DataFrame(
    {
        "eta": [eta],
        "fguide_g": [fguide],
        "fjoint_g": [fjoint],
        "r2": [r2],
    }
)

results.to_csv(
    OUTPUT_DIR / "fitted_parameters.csv",
    index=False,
)

save_fit_summary(
    eta,
    fguide,
    fjoint,
    r2,
)


# =========================================================
# Save Predictions
# =========================================================

fit_df.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False,
)


# =========================================================
# Global Measured vs Model
# =========================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    fit_df["F_out"],
    fit_df["prediction"],
)

mn = min(
    fit_df["F_out"].min(),
    fit_df["prediction"].min(),
)

mx = max(
    fit_df["F_out"].max(),
    fit_df["prediction"].max(),
)

plt.plot(
    [mn, mx],
    [mn, mx],
)

plt.xlabel("Measured F_out [g]")
plt.ylabel("Predicted F_out [g]")
plt.title("Measured vs Model")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "measured_vs_model_global.png",
    dpi=300,
)

plt.close()


# =========================================================
# Residual Histogram
# =========================================================

plt.figure(figsize=(7, 5))

plt.hist(
    fit_df["residual"],
    bins=25,
)

plt.xlabel("Residual [g]")
plt.ylabel("Count")
plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "residual_histogram.png",
    dpi=300,
)

plt.close()


# =========================================================
# Residual vs l_offset
# =========================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    fit_df["l_offset"],
    fit_df["residual"],
)

plt.axhline(
    0,
    linestyle="--",
)

plt.xlabel("l_offset [mm]")
plt.ylabel("Residual [g]")
plt.title("Residual vs l_offset")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "residual_vs_loffset.png",
    dpi=300,
)

plt.close()


# =========================================================
# Efficiency vs n
# =========================================================

n_values = np.sort(
    metrics["n_units"].unique()
)

eta_model = eta ** n_values

plt.figure(figsize=(7, 5))

plt.plot(
    n_values,
    eta_model,
    marker="o",
)

plt.xlabel("n_units")
plt.ylabel("eta^n")
plt.title("Predicted Transmission Efficiency")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "efficiency_vs_n.png",
    dpi=300,
)

plt.close()


# =========================================================
# Friction vs n
# =========================================================

friction_model = (
    fguide
    + fjoint * n_values
)

plt.figure(figsize=(7, 5))

plt.plot(
    n_values,
    friction_model,
    marker="o",
)

plt.xlabel("n_units")
plt.ylabel("Friction [g]")
plt.title("Predicted Friction")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "friction_vs_n.png",
    dpi=300,
)

plt.close()


# =========================================================
# Delta F Validation
# =========================================================

if "delta_F_g" in metrics.columns:

    predicted_delta = (
        2
        * (
            fguide
            + fjoint * metrics["n_units"]
        )
    )

    plt.figure(figsize=(7, 5))

    plt.scatter(
        metrics["delta_F_g"],
        predicted_delta,
    )

    mn = min(
        metrics["delta_F_g"].min(),
        predicted_delta.min(),
    )

    mx = max(
        metrics["delta_F_g"].max(),
        predicted_delta.max(),
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
    )

    plt.xlabel("Measured ΔF [g]")
    plt.ylabel("Predicted ΔF [g]")
    plt.title("Friction Model Validation")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "deltaF_validation.png",
        dpi=300,
    )

    plt.close()


print("\n====================================")
print("Regression complete.")
print(f"Results saved to:\n{OUTPUT_DIR}")
print("====================================")
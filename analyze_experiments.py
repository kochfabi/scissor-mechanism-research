"""
compare_experiments.py
----------------------
Comparative analysis of scissor gripper force transmission experiments.

Usage:
    python compare_experiments.py <folder_path>

Scans <folder_path> recursively for all files named summary.csv,
parses each one, and produces:
  1. Terminal statistics table
  2. Efficiency (ε) vs F_in comparison plot  [plot_epsilon.png]
  3. Hysteresis (H%) vs F_in comparison plot [plot_hysteresis.png]
  4. Exported metrics CSV                     [metrics.csv]

All outputs are saved to <folder_path>/analysis/.
"""

import sys
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── 1. Parsing ────────────────────────────────────────────────────────────────

def parse_summary(path: str) -> dict | None:
    """
    Parse a summary.csv produced by experiment.py.

    Returns a dict with:
      meta   – dict of metadata fields (title, l_offset, n_units, l_curve, …)
      trials – list of dicts, one per non-zero-F_in trial
      zeros  – list of dicts for F_in = 0 trials (residual measurements)
    Returns None if the file cannot be parsed.
    """
    meta = {}
    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_found = False
        col_names = []

        for row in reader:
            if not row:
                continue

            # Metadata rows start with '#'
            if row[0].startswith("#"):
                key = row[0].lstrip("#").strip()
                value = row[1].strip() if len(row) > 1 else ""
                meta[key] = value
                continue

            # Column header row
            if not header_found:
                col_names = [c.strip() for c in row]
                header_found = True
                continue

            # Data rows
            if header_found and col_names:
                try:
                    entry = {}
                    for k, v in zip(col_names, row):
                        v = v.strip()
                        if v == "":
                            entry[k] = None
                        else:
                            try:
                                entry[k] = float(v)
                            except ValueError:
                                entry[k] = v
                    rows.append(entry)
                except Exception:
                    continue

    if not rows:
        return None

    # Separate zero-load and non-zero-load trials
    trials = [r for r in rows if r.get("F_in") not in (None, 0.0)]
    zeros  = [r for r in rows if r.get("F_in") in (None, 0.0)]

    # Coerce numeric fields
    for r in trials + zeros:
        for field in ("F_in", "mean_force_g", "std_force_g", "epsilon", "n_samples"):
            if field in r and r[field] is not None:
                try:
                    r[field] = float(r[field])
                except (ValueError, TypeError):
                    r[field] = None

    return {"meta": meta, "trials": trials, "zeros": zeros, "path": path}


# ── 2. Run segmentation ───────────────────────────────────────────────────────

def segment_runs(trials: list) -> list[dict]:
    """
    Split a flat trial list into loading/unloading segments.

    A new segment starts whenever:
      - The direction of F_in changes (ascending → descending or vice versa)

    Returns a list of segments, each a dict:
      direction  – "loading" or "unloading"
      trials     – list of trial dicts in this segment
    """
    if not trials:
        return []

    segments = []
    current_trials = [trials[0]]
    current_dir = None  # unknown until second point

    for i in range(1, len(trials)):
        prev_fin = trials[i - 1]["F_in"]
        curr_fin = trials[i]["F_in"]

        if curr_fin is None or prev_fin is None:
            continue

        new_dir = "loading" if curr_fin >= prev_fin else "unloading"

        if current_dir is None:
            current_dir = new_dir

        if new_dir != current_dir:
            # Direction reversed → close current segment, start new one
            segments.append({"direction": current_dir, "trials": current_trials})
            current_trials = [trials[i]]
            current_dir = new_dir
        else:
            current_trials.append(trials[i])

    # Close final segment
    if current_trials:
        segments.append({"direction": current_dir or "loading", "trials": current_trials})

    return segments


# ── 3. Metrics computation ────────────────────────────────────────────────────

def compute_metrics(dataset: dict) -> dict:
    """
    Compute per-F_in metrics from a parsed dataset.

    Returns a dict with keys:
      label           – human-readable label from metadata
      loading         – dict {F_in: {"epsilon", "mean_force_g", "std_force_g"}}
      unloading       – dict {F_in: {"epsilon", "mean_force_g", "std_force_g"}}
      hysteresis      – dict {F_in: {"delta_F_g", "H_pct"}}   (only where both paths exist)
      mean_eps_load   – float
      mean_eps_unload – float
      zero_residuals  – list of mean_force_g from zero-load trials
      mean_zero       – float (mean of zero residuals, or None)
      meta            – original metadata dict
    """
    meta   = dataset["meta"]
    trials = dataset["trials"]
    zeros  = dataset["zeros"]
    segs   = segment_runs(trials)

    # Aggregate loading/unloading: average across all runs at each F_in level
    def aggregate(direction: str) -> dict:
        buckets = {}  # F_in → list of mean_force_g values
        eps_buckets = {}
        for seg in segs:
            if seg["direction"] != direction:
                continue
            for t in seg["trials"]:
                fin = t["F_in"]
                if fin is None:
                    continue
                buckets.setdefault(fin, []).append(t["mean_force_g"])
                if t["epsilon"] is not None:
                    eps_buckets.setdefault(fin, []).append(t["epsilon"])

        result = {}
        for fin, forces in buckets.items():
            result[fin] = {
                "mean_force_g": float(np.mean(forces)),
                "epsilon":      float(np.mean(eps_buckets[fin])) if fin in eps_buckets else None,
            }
        return result

    loading   = aggregate("loading")
    unloading = aggregate("unloading")

    # Hysteresis: only where both paths exist at the same F_in
    hysteresis = {}
    for fin in loading:
        if fin in unloading and loading[fin]["mean_force_g"] is not None and unloading[fin]["mean_force_g"] is not None:
            delta = unloading[fin]["mean_force_g"] - loading[fin]["mean_force_g"]
            hysteresis[fin] = {
                "delta_F_g": delta,
                "H_pct":     delta / fin if fin != 0 else None,
            }

    mean_eps_load   = float(np.mean([v["epsilon"] for v in loading.values()   if v["epsilon"] is not None])) if loading   else None
    mean_eps_unload = float(np.mean([v["epsilon"] for v in unloading.values() if v["epsilon"] is not None])) if unloading else None

    zero_residuals = [z["mean_force_g"] for z in zeros if z.get("mean_force_g") is not None]
    mean_zero      = float(np.mean(zero_residuals)) if zero_residuals else None

    # Build a label from metadata
    l_offset = meta.get("l_offset", "?")
    n_units  = meta.get("n_units",  "?")
    l_curve  = meta.get("l_curve",  "?")
    title    = meta.get("title",    "")
    label    = f"l_off={l_offset}, n={n_units}, l_c={l_curve}"
    if title and title.lower() != "untitled":
        label = f"{title} | {label}"

    return {
        "label":           label,
        "loading":         loading,
        "unloading":       unloading,
        "hysteresis":      hysteresis,
        "mean_eps_load":   mean_eps_load,
        "mean_eps_unload": mean_eps_unload,
        "zero_residuals":  zero_residuals,
        "mean_zero":       mean_zero,
        "meta":            meta,
    }


# ── 4. Terminal statistics ────────────────────────────────────────────────────

def print_statistics(all_metrics: list[dict]):
    sep = "─" * 80

    for m in all_metrics:
        print(f"\n{'═' * 80}")
        print(f"  {m['label']}")
        print(f"  File: {m['meta'].get('_path', '')}")
        print(f"{'═' * 80}")

        # Zero-load residuals
        if m["zero_residuals"]:
            resids = m["zero_residuals"]
            print(f"\n  Zero-load residuals: {[f'{r:.2f}' for r in resids]} g")
            print(f"  Mean residual: {m['mean_zero']:.3f} g   σ = {float(np.std(resids)):.3f} g")

        # Loading path
        print(f"\n  {'F_in':>8}  {'F_out load':>12}  {'ε load':>8}  {'F_out unload':>13}  {'ε unload':>10}  {'ΔF':>8}  {'H%':>7}")
        print(f"  {sep}")
        all_fins = sorted(set(list(m["loading"].keys()) + list(m["unloading"].keys())))
        for fin in all_fins:
            ld = m["loading"].get(fin)
            ul = m["unloading"].get(fin)
            hy = m["hysteresis"].get(fin)

            fl  = f"{ld['mean_force_g']:>12.2f}"  if ld  else f"{'—':>12}"
            el  = f"{ld['epsilon']:>8.4f}"         if ld and ld["epsilon"] else f"{'—':>8}"
            fu  = f"{ul['mean_force_g']:>13.2f}"  if ul  else f"{'—':>13}"
            eu  = f"{ul['epsilon']:>10.4f}"        if ul and ul["epsilon"] else f"{'—':>10}"
            dF  = f"{hy['delta_F_g']:>8.2f}"       if hy  else f"{'—':>8}"
            hp  = f"{hy['H_pct']:>6.1f}%"          if hy and hy["H_pct"] is not None else f"{'—':>7}"

            print(f"  {fin:>8.0f}  {fl}  {el}  {fu}  {eu}  {dF}  {hp}")

        # Summary
        print(f"\n  Mean ε loading:   {m['mean_eps_load']:.4f}" if m["mean_eps_load"]   else "")
        print(f"  Mean ε unloading: {m['mean_eps_unload']:.4f}" if m["mean_eps_unload"] else "")
        if m["mean_eps_load"] and m["mean_eps_unload"]:
            print(f"  Δε (unload−load): {m['mean_eps_unload'] - m['mean_eps_load']:+.4f}")

    print(f"\n{'═' * 80}\n")


# ── 5. Plots ──────────────────────────────────────────────────────────────────

def _colors(n: int):
    cmap = plt.colormaps["tab10"]
    return [cmap(i % 10) for i in range(n)]


def plot_epsilon(all_metrics: list[dict], output_dir: str):
    colors = _colors(len(all_metrics))

    # Efficiency figure
    fig_eff, ax_eff = plt.subplots(figsize=(10, 6))
    for m, color in zip(all_metrics, colors):
        label = m["label"]
        fins_l = sorted(m["loading"].keys())
        eps_l = [m["loading"][f]["epsilon"] for f in fins_l if m["loading"][f]["epsilon"] is not None]
        fins_l = [f for f in fins_l if m["loading"][f]["epsilon"] is not None]
        if fins_l:
            ax_eff.plot(fins_l, eps_l, marker="o", markersize=4, linestyle="-",
                        color=color, linewidth=1.2, label=f"{label} [load]")
            for i in range(len(fins_l) - 1):
                mid_fin = (fins_l[i] + fins_l[i+1]) / 2
                mid_eps = (eps_l[i] + eps_l[i+1]) / 2
                dx = (fins_l[i+1] - fins_l[i]) * 0.01
                dy = (eps_l[i+1] - eps_l[i]) * 0.01
                ax_eff.annotate("", xy=(mid_fin + dx, mid_eps + dy), xytext=(mid_fin - dx, mid_eps - dy),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1, alpha=0.8))

        fins_u = sorted(m["unloading"].keys())
        eps_u = [m["unloading"][f]["epsilon"] for f in fins_u if m["unloading"][f]["epsilon"] is not None]
        fins_u = [f for f in fins_u if m["unloading"][f]["epsilon"] is not None]
        if fins_u:
            ax_eff.plot(fins_u, eps_u, marker="s", markersize=4, linestyle="--",
                        color=color, linewidth=1.2, label=f"{label} [unload]")
            for i in range(len(fins_u) - 1):
                mid_fin = (fins_u[i] + fins_u[i+1]) / 2
                mid_eps = (eps_u[i] + eps_u[i+1]) / 2
                dx = (fins_u[i+1] - fins_u[i]) * 0.01
                dy = (eps_u[i+1] - eps_u[i]) * 0.01
                ax_eff.annotate("", xy=(mid_fin - dx, mid_eps - dy), xytext=(mid_fin + dx, mid_eps + dy),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1, alpha=0.8))

    all_fins_all = sorted({f for m in all_metrics for f in list(m["loading"].keys()) + list(m["unloading"].keys())})
    if all_fins_all:
        ax_eff.axhline(1.0, color="red", linestyle=":", linewidth=1.2, label="Ideal (ε = 1)")

    ax_eff.set_xlabel("F_in [g]", fontsize=12)
    ax_eff.set_ylabel("ε = F_out / F_in", fontsize=12)
    ax_eff.set_title("Force Transmission Efficiency", fontsize=13)
    ax_eff.grid(True, linestyle="--", alpha=0.4)
    ax_eff.set_ylim(bottom=0.7)
    ax_eff.legend(fontsize=8, framealpha=0.6)

    out_eff = os.path.join(output_dir, "plot_epsilon_efficiency.png")
    plt.tight_layout()
    plt.savefig(out_eff, dpi=150, bbox_inches="tight")
    plt.close(fig_eff)
    print(f"  Saved → {out_eff}")

    # Force difference figure (F_out − F_in)
    fig_fd, ax_fd = plt.subplots(figsize=(10, 6))
    for m, color in zip(all_metrics, colors):
        label = m["label"]
        fins_l = sorted(m["loading"].keys())
        fins_l = [f for f in fins_l if m["loading"][f]["epsilon"] is not None]
        if fins_l:
            diff_l = [m["loading"][f]["mean_force_g"] - f for f in fins_l]
            ax_fd.plot(fins_l, diff_l, marker="o", markersize=4, linestyle="-",
                       color=color, linewidth=1.2, label=f"{label} [load]")
            for i in range(len(fins_l) - 1):
                mid_fin = (fins_l[i] + fins_l[i+1]) / 2
                mid_diff = (diff_l[i] + diff_l[i+1]) / 2
                dx = (fins_l[i+1] - fins_l[i]) * 0.01
                dy = (diff_l[i+1] - diff_l[i]) * 0.01
                ax_fd.annotate("", xy=(mid_fin + dx, mid_diff + dy), xytext=(mid_fin - dx, mid_diff - dy),
                               arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.8))

        fins_u = sorted(m["unloading"].keys())
        fins_u = [f for f in fins_u if m["unloading"][f]["epsilon"] is not None]
        if fins_u:
            diff_u = [m["unloading"][f]["mean_force_g"] - f for f in fins_u]
            ax_fd.plot(fins_u, diff_u, marker="s", markersize=4, linestyle="--",
                       color=color, linewidth=1.2, label=f"{label} [unload]")
            for i in range(len(fins_u) - 1):
                mid_fin = (fins_u[i] + fins_u[i+1]) / 2
                mid_diff = (diff_u[i] + diff_u[i+1]) / 2
                dx = (fins_u[i+1] - fins_u[i]) * 0.01
                dy = (diff_u[i+1] - diff_u[i]) * 0.01
                ax_fd.annotate("", xy=(mid_fin - dx, mid_diff - dy), xytext=(mid_fin + dx, mid_diff + dy),
                               arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, alpha=0.5))

    ax_fd.axhline(0.0, color="red", linestyle=":", linewidth=1.0, label="Ideal (F_out = F_in)")
    ax_fd.set_xlabel("F_in [g]", fontsize=12)
    ax_fd.set_ylabel("F_out − F_in [g]", fontsize=12)
    ax_fd.set_title("Force Difference", fontsize=13)
    ax_fd.grid(True, linestyle="--", alpha=0.4)
    ax_fd.legend(fontsize=8, framealpha=0.6)
    out_fd = os.path.join(output_dir, "plot_force_difference.png")
    plt.tight_layout()
    plt.savefig(out_fd, dpi=150, bbox_inches="tight")
    plt.close(fig_fd)
    print(f"  Saved → {out_fd}")


def plot_hysteresis(all_metrics: list[dict], output_dir: str):
    colors = _colors(len(all_metrics))

    # Absolute ΔF plot
    fig_a, ax_a = plt.subplots(figsize=(10, 6))
    for m, color in zip(all_metrics, colors):
        label = m["label"]
        fins = sorted(m["hysteresis"].keys())
        if not fins:
            continue
        delta_F = [m["hysteresis"][f]["delta_F_g"] for f in fins]
        ax_a.plot(fins, delta_F, marker="o", markersize=5, linestyle="-", color=color, linewidth=1.5, label=label)
    ax_a.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax_a.set_xlabel("F_in [g]", fontsize=12)
    ax_a.set_ylabel("ΔF = F_out(unload) − F_out(load)  [g]", fontsize=11)
    ax_a.set_title("Absolute Hysteresis", fontsize=12)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, linestyle="--", alpha=0.4)
    out_a = os.path.join(output_dir, "plot_hysteresis_absolute.png")
    plt.tight_layout()
    plt.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print(f"  Saved → {out_a}")

    # Normalised hysteresis ratio H = ΔF / F_in (not percent)
    fig_r, ax_r = plt.subplots(figsize=(10, 6))
    for m, color in zip(all_metrics, colors):
        label = m["label"]
        fins = sorted(m["hysteresis"].keys())
        fins_r = [f for f in fins if f != 0 and m["hysteresis"][f]["delta_F_g"] is not None]
        if fins_r:
            ratio = [m["hysteresis"][f]["delta_F_g"] / f for f in fins_r]
            ax_r.plot(fins_r, ratio, marker="o", markersize=5, linestyle="-", color=color, linewidth=1.5, label=label)
    ax_r.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax_r.set_xlabel("F_in [g]", fontsize=12)
    ax_r.set_ylabel("H = ΔF / F_in", fontsize=11)
    ax_r.set_title("Normalised Hysteresis Ratio", fontsize=12)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, linestyle="--", alpha=0.4)
    out_r = os.path.join(output_dir, "plot_hysteresis_ratio.png")
    plt.tight_layout()
    plt.savefig(out_r, dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    print(f"  Saved → {out_r}")


# ── 6. CSV export ─────────────────────────────────────────────────────────────

def export_metrics_csv(all_metrics: list[dict], output_dir: str):
    out = os.path.join(output_dir, "metrics.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Per-F_in detail table
        writer.writerow(["# Per-condition metrics"])
        writer.writerow([
            "label", "l_offset", "n_units", "l_curve",
            "F_in_g",
            "F_out_load_g", "epsilon_load",
            "F_out_unload_g", "epsilon_unload",
            "delta_F_g", "H_pct",
        ])

        for m in all_metrics:
            meta = m["meta"]
            lo   = meta.get("l_offset", "")
            nu   = meta.get("n_units",  "")
            lc   = meta.get("l_curve",  "")
            all_fins = sorted(set(list(m["loading"].keys()) + list(m["unloading"].keys())))

            for fin in all_fins:
                ld = m["loading"].get(fin)
                ul = m["unloading"].get(fin)
                hy = m["hysteresis"].get(fin)
                writer.writerow([
                    m["label"], lo, nu, lc, fin,
                    f"{ld['mean_force_g']:.4f}" if ld else "",
                    f"{ld['epsilon']:.6f}"       if ld and ld["epsilon"] else "",
                    f"{ul['mean_force_g']:.4f}" if ul else "",
                    f"{ul['epsilon']:.6f}"       if ul and ul["epsilon"] else "",
                    f"{hy['delta_F_g']:.4f}"    if hy else "",
                    f"{hy['H_pct']:.3f}"        if hy and hy["H_pct"] is not None else "",
                ])

        # Summary table
        writer.writerow([])
        writer.writerow(["# Summary per configuration"])
        writer.writerow([
            "label", "l_offset", "n_units", "l_curve", "mean_eps_load", "mean_eps_unload", "delta_eps", "mean_delta_F_g", "mean_zero_residual_g",
        ])
        for m in all_metrics:
            meta = m["meta"]
            de = (m["mean_eps_unload"] - m["mean_eps_load"]) if (m["mean_eps_load"] and m["mean_eps_unload"]) else ""
            dF = np.mean([hy["delta_F_g"] for hy in m["hysteresis"].values()]) if m["hysteresis"] else ""
            writer.writerow([
                m["label"],
                meta.get("l_offset", ""), meta.get("n_units", ""), meta.get("l_curve", ""),
                f"{m['mean_eps_load']:.6f}"   if m["mean_eps_load"]   else "",
                f"{m['mean_eps_unload']:.6f}" if m["mean_eps_unload"] else "",
                f"{de:.6f}"                   if de != ""             else "",
                f"{dF:.4f}"                   if dF != ""             else "",
                f"{m['mean_zero']:.4f}"       if m["mean_zero"] is not None else "",
            ])

    print(f"  Saved → {out}")


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_experiments.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    # Find all summary.csv files recursively
    pattern = os.path.join(folder, "**", "summary.csv")
    paths   = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        print(f"No summary.csv files found under '{folder}'.")
        sys.exit(1)

    print(f"\nFound {len(paths)} summary file(s):")
    for p in paths:
        print(f"  {p}")

    # Parse and compute
    all_metrics = []
    for p in paths:
        dataset = parse_summary(p)
        if dataset is None:
            print(f"  WARNING: Could not parse {p}, skipping.")
            continue
        dataset["meta"]["_path"] = p
        m = compute_metrics(dataset)
        m["meta"]["_path"] = p
        all_metrics.append(m)

    if not all_metrics:
        print("No valid datasets found. Exiting.")
        sys.exit(1)

    # Output directory
    output_dir = os.path.join(folder, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Run outputs
    print_statistics(all_metrics)
    print("Generating plots...")
    plot_epsilon(all_metrics, output_dir)
    plot_hysteresis(all_metrics, output_dir)
    print("Exporting metrics CSV...")
    export_metrics_csv(all_metrics, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

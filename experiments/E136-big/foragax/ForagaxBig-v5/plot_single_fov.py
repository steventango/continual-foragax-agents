"""Plot learning curve for a single FOV (aggregated across all seeds)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

try:
    from PyExpPlotting.matplot import save, setDefaultConference, setFonts
    setDefaultConference("jmlr")
    setFonts(20)
except ImportError:
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / EXP_DIR.relative_to(ROOT / "experiments")
PARQUET = RESULTS_DIR / "data.parquet"
PLOT_DIR = EXP_DIR / "plots"

FOV = 9
AGENT = "DQN"
METRIC = "ewm_reward"

COLORS = {
    "DQN": "#8b8cff",
}

def main():
    if not PARQUET.exists():
        print(f"Parquet file not found: {PARQUET}")
        return

    df = pl.read_parquet(PARQUET)
    # Cast Float16 columns to Float32 — Polars aggregations don't support Float16.
    df = df.with_columns(pl.col(pl.Float16).cast(pl.Float32))

    # Filter for the specific FOV and agent
    filtered_df = df.filter((pl.col("aperture") == FOV) & (pl.col("alg") == AGENT))

    if filtered_df.height == 0:
        print(f"No data found for {AGENT} with FOV {FOV}")
        return

    # Average over seeds at each frame, ±1.96 SEM band
    agg = (
        filtered_df.group_by("frame")
        .agg(
            pl.col(METRIC).mean().alias("mean"),
            pl.col(METRIC).std().alias("std"),
            pl.col(METRIC).count().alias("n"),
        )
        .sort("frame")
        .drop_nulls()
    )

    frames = agg["frame"].to_numpy()
    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()
    ns = np.maximum(agg["n"].to_numpy(), 1)
    sem = stds / np.sqrt(ns)
    lower = means - 1.96 * sem
    upper = means + 1.96 * sem

    fig, ax = plt.subplots(figsize=(10, 6))

    label = f"{AGENT}-{FOV}"
    ax.plot(frames, means, label=label, color=COLORS[AGENT], linewidth=0.5)
    ax.fill_between(
        frames, lower, upper, color=COLORS[AGENT], alpha=0.2, linewidth=0
    )

    ax.set_xlabel("Time steps")
    ax.set_ylabel("Average Reward")
    ax.set_ylim(0, 0.05)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"learning_curve_{AGENT}_{FOV}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {out}")

if __name__ == "__main__":
    main()

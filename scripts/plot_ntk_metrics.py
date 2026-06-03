import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # Go up from scripts/ to repo root
sys.path.insert(0, str(ROOT / "src"))

# Correct path to your results
DQN_TWO_BIOME_LARGE_V1 = "results/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN/data"
DQN_SQUARE_WAVE_TWO_BIOME_V11 = "results/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN/data"
DQN_FORAGAX_BIG_V5 = "results/E136-big/foragax/ForagaxBig-v5/9/DQN/data"

PPO_SQUARE_WAVE_TWO_BIOME_V11 = "results/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/ActorCriticMLP/data"

data_path = ROOT / PPO_SQUARE_WAVE_TWO_BIOME_V11

# Each panel lists every column that could supply it, paired with a series
# label.  DQN runs write `churn_norm` / `ntk_rank` / `ntk_cond`; PPO runs write
# separate `value_*` / `policy_*` columns.  Only the columns actually present in
# the loaded data are plotted, so this works unchanged for either agent (or a
# mix), drawing one line per available series.
PANELS = [
    {
        "ylabel": "Churn Norm",
        "title": "Churn Over Time",
        "log": False,
        "series": [
            ("churn_norm", "DQN"),
            ("value_churn", "Value (PPO)"),
            ("policy_churn", "Policy (PPO)"),
        ],
    },
    {
        "ylabel": "NTK Rank",
        "title": "NTK Rank Over Time",
        "log": False,
        "series": [
            ("ntk_rank", "DQN"),
            ("value_ntk_rank", "Value (PPO)"),
            ("policy_ntk_rank", "Policy (PPO)"),
        ],
    },
    {
        "ylabel": "NTK Condition Number",
        "title": "NTK Condition Number Over Time",
        "log": True,  # condition numbers span many orders of magnitude
        "series": [
            ("ntk_cond", "DQN"),
            ("value_ntk_cond", "Value (PPO)"),
            ("policy_ntk_cond", "Policy (PPO)"),
        ],
    },
]

print(f"Data path: {data_path}")
print(f"Data path exists: {data_path.exists()}")

# Load the raw per-run npz files.  We read these directly (rather than via
# read_metrics_from_data) because the metrics are stored at their native
# resolution here: DQN writes one value per env step (NaN except every
# ntk_freq), while PPO writes one value per *update*.  The reader would repeat
# the PPO per-update arrays up to per-step length, which makes constant-valued
# series (e.g. NTK rank) indistinguishable from a single repeated point.
runs = []
for f in sorted(Path(data_path).glob("*.npz")):
    with np.load(f) as d:
        runs.append({k: np.asarray(d[k]) for k in d.keys()})
print(f"Loaded {len(runs)} run(s)")


def metric_series(col: str):
    """Seed-averaged (x_steps, values) for `col`, or None if absent/empty.

    A metric array has one entry per *measurement* (per env step for DQN, per
    update for PPO).  Its x-axis is recovered in env steps by comparing its
    length to the per-step `rewards` array: `steps_per_point = len(rewards) /
    len(metric)` (1 for DQN, rollout_steps for PPO).  NaN entries (non-metric
    steps/updates) are dropped *after* averaging across seeds.
    """
    arrays = []
    steps_per_point = 1.0
    for r in runs:
        if col not in r:
            continue
        v = np.asarray(r[col], dtype=float).reshape(-1)
        if v.shape[0] == 0:
            continue
        base_len = r["rewards"].reshape(-1).shape[0] if "rewards" in r else v.shape[0]
        steps_per_point = base_len / v.shape[0]
        arrays.append(v)

    if not arrays:
        return None

    # Align seeds to the shortest run, average (ignoring NaN measurements).
    m = min(a.shape[0] for a in arrays)
    stacked = np.stack([a[:m] for a in arrays], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        mean = np.nanmean(stacked, axis=0)

    x = np.arange(m) * steps_per_point
    finite = np.isfinite(mean)
    if not finite.any():
        return None
    return x[finite], mean[finite]


# Report which metric columns are present and have computed (finite) values.
all_metric_cols = [col for panel in PANELS for col, _ in panel["series"]]
present_cols = [c for c in all_metric_cols if any(c in r for r in runs)]
for metric in all_metric_cols:
    series = metric_series(metric)
    if metric not in present_cols:
        print(f"{metric}: NOT FOUND in data")
    elif series is None:
        print(f"{metric}: present but no finite values")
    else:
        x, y = series
        print(f"{metric}: {len(y)} finite point(s) | "
              f"min={y.min():.4g} max={y.max():.4g} mean={y.mean():.4g}")

if all(metric_series(c) is None for c in present_cols):
    print("\nWARNING: No finite metric values found!")
    print("The metrics may not have been computed during training.")
    exit(1)

# Plot
fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4))
if len(PANELS) == 1:
    axes = [axes]

for ax, panel in zip(axes, PANELS):
    n_plotted = 0
    for col, label in panel["series"]:
        series = metric_series(col)
        if series is None:
            continue
        x, y = series
        ax.plot(x, y, marker="o", label=label)
        n_plotted += 1

    ax.set_xlabel("Step")
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"])
    if panel["log"] and n_plotted > 0:
        ax.set_yscale("log")
    # Legend only matters when multiple series share a panel (e.g. PPO value vs
    # policy); a single DQN line doesn't need one.
    if n_plotted > 1:
        ax.legend()

plt.tight_layout()
plt.savefig("ntk_metrics.png")
plt.show()

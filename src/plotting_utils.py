import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import tol_colors as tc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def save(save_path: str, plot_name: str, save_type: str, f: Figure, **kwargs):
    """Save matplotlib figure to file."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / f"{plot_name}.{save_type}"
    f.savefig(file_path, bbox_inches="tight", **kwargs)
    print(f"Plot saved to: {file_path}")


sys.path.append(os.getcwd())

# ---------------------------------
# Matplotlib and Seaborn Settings
# ---------------------------------
sns.set_palette(tc.tol_cset("vibrant"))

# Set font sizes for better readability in papers
fontsize = 14
plt.rcParams["axes.labelsize"] = fontsize  # Axis labels
plt.rcParams["xtick.labelsize"] = fontsize - 2  # X-tick labels
plt.rcParams["ytick.labelsize"] = fontsize - 2  # Y-tick labels

# ---------------------
# Constants
# ---------------------
LABEL_MAP: Dict[str, str] = {
    "DQN": "DQN",
    "DQN_CReLU": "DQN (CReLU)",
    "DQN_L2": "DQN (L2)",
    "DQN_L2_Init": "DQN (L2 Init)",
    "DQN_LN": "DQN (LayerNorm)",
    "DQN_Reset_Head": "DQN (Head Reset)",
    "DQN_Hare_and_Tortoise": "DQN (Hare & Tortoise)",
    "DQN_Shrink_and_Perturb": "DQN (Shrink & Perturb)",
    "DQN_privileged": "DQN (Privileged)",
    "DQN_world": "DQN (World)",
    "Search-Brown": "Search (Brown)",
    "Search-Brown-Avoid-Green": "Search (Brown, Avoid Green)",
    "Search-Morel": "Search (Morel)",
    "Search-Morel-Avoid-Green": "Search (Morel, Avoid Green)",
    "Search-Nearest": "Search (Nearest)",
    "Search-Oracle": "Search (Oracle)",
    "Search-Oyster": "Search (Oyster)",
}

frozen_label_map = {}
for key in list(LABEL_MAP.keys()):
    frozen_label_map[f"{key}_greedy_frozen_5M"] = (
        f"{LABEL_MAP[key]} (Greedy Frozen @ 5 M)"
    )
    frozen_label_map[f"{key}_greedy_frozen_1M"] = (
        f"{LABEL_MAP[key]} (Greedy Frozen @ 1 M)"
    )
    frozen_label_map[f"{key}_frozen_1M"] = f"{LABEL_MAP[key]} (Frozen @ 1 M)"
    frozen_label_map[f"{key}_frozen_5M"] = f"{LABEL_MAP[key]} (Frozen @ 5 M)"
LABEL_MAP.update(frozen_label_map)


YLABEL_MAP: Dict[str, str] = {
    "Ewm Reward": "Average Reward",
}


# Color scheme for plotting
colorset = tc.tol_cset("high_contrast")
sunset_colormap = tc.tol_cmap("sunset")

# Biome colors for plotting
TWO_BIOME_COLORS: Dict[str, Any] = {
    "Morel": colorset.red,
    "Oyster": colorset.yellow,
    "Neither": colorset.blue,
}

WEATHER_BIOME_COLORS: Dict[str, Any] = {
    "Cold": sunset_colormap(0.0),
    "Neither": sunset_colormap(0.5),
    "Hot": sunset_colormap(1.0),
}


# ---------------------
# Mappings
# ---------------------
def get_biome_mapping(env: str) -> Dict[int, str]:
    if "TwoBiome" in env:
        return {-1: "Neither", 0: "Morel", 1: "Oyster"}
    if "Weather" in env:
        return {-1: "Neither", 0: "Hot", 1: "Cold"}
    raise ValueError(f"Unknown biome mapping for environment: {env}")


def get_object_mapping(env: str) -> Dict[int, str]:
    if "Foragax" in env:
        return {0: "Morel", 1: "Oyster", 2: "Chantrelle"}
    raise ValueError(f"Unknown object mapping for environment: {env}")


# ---------------------
# Data Loading
# ---------------------
def load_data(experiment_path: Path) -> pl.DataFrame:
    """Loads and preprocesses data from the specified experiment_path."""
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
        / "data.parquet"
    )
    df = pl.read_parquet(data_path)
    df = df.with_columns(
        pl.col("alg").str.replace(r"_frozen_.*", "").alias("alg_base"),
        pl.when(pl.col("alg").str.contains("_frozen"))
        .then(pl.col("alg").str.extract(r"_frozen_(.*)", 1))
        .otherwise(None)
        .alias("freeze_steps_str"),
    )
    return df


# ---------------------
# Data Filtering
# ---------------------
def filter_by_alg_aperture(
    df: pl.DataFrame, filter_strings: Optional[List[str]]
) -> pl.DataFrame:
    if not filter_strings:
        return df

    conditions = []
    for faa in filter_strings:
        parts = faa.split(":")
        alg = parts[0]

        cond = pl.col("alg") == alg
        if len(parts) > 1:
            aperture = int(parts[1])
            cond = cond & (pl.col("aperture") == aperture)

        conditions.append(cond)

    return df.filter(pl.any_horizontal(conditions))


# ---------------------
# Argument Parsing
# ---------------------
class PlottingArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("path", type=str, help="Path to the experiment directory")
        self.add_argument(
            "--save-type", type=str, default="pdf", help="File format for saving plots"
        )
        self.add_argument(
            "--plot-name", type=str, default=None, help="Custom plot name"
        )
        self.add_argument(
            "--filter-algs",
            type=str,
            nargs="*",
            help="Algorithms to include in the plot",
        )
        self.add_argument(
            "--filter-seeds", type=int, nargs="*", help="Seeds to include in the plot"
        )
        self.add_argument(
            "--filter-alg-apertures",
            nargs="*",
            help="Filter for specific algorithm-aperture pairs (e.g., 'DQN:9').",
        )


def parse_plotting_args(parser: PlottingArgumentParser) -> argparse.Namespace:
    """Parses and returns common plotting arguments."""
    args = parser.parse_args()
    args.experiment_path = Path(args.path).resolve()
    return args


# ---------------------
# Plot Saving
# ---------------------
def save_plot(
    fig: Figure,
    experiment_path: Path,
    plot_name: str,
    save_type: str,
    **kwargs: Any,
):
    """Saves a matplotlib figure to the specified path."""
    save_path = experiment_path / "plots"
    save(
        save_path=str(save_path),
        plot_name=plot_name,
        save_type=save_type,
        f=fig,
        **kwargs,
    )


# ---------------------
# Plotting Utilities
# ---------------------
def format_metric_name(metric: str) -> str:
    """Formats a metric name for display."""
    return metric.replace("_", " ").title()


def get_mapped_label(label: str, label_map: Optional[Dict[str, str]] = None) -> str:
    """Get the mapped label, handling apertures and frozen variants."""
    if label_map and label in label_map:
        return label_map[label]
    if ":" in label:
        alg, aperture = label.split(":", 1)
        base_label = label_map.get(alg, alg) if label_map else alg
        return f"{base_label} (FOV {aperture})"
    return label


def get_legend_elements(labels: List[str], colors: Dict[str, Any]) -> List[Line2D]:
    """Creates legend elements for a plot."""
    return [
        Line2D([0], [0], color=colors[label], lw=2, label=LABEL_MAP.get(label, label))
        for label in labels
    ]


def despine(ax: Axes):
    """Removes top and right spines from a plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

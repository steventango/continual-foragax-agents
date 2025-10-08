import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from utils.constants import LABEL_MAP, TWO_BIOME_COLORS, WEATHER_BIOME_COLORS
from utils.plotting import select_colors

setDefaultConference("jmlr")
setFonts(20)

SAMPLE_TYPE_MAP = {
    "999000:1000000:500": "Early learning",
    "4999000:5000000:500": "Mid learning",
    "9999000:10000000:500": "Late learning",
}


def main(
    experiment_path: Path,
    save_type: str = "pdf",
    sample_types: list[str] | None = None,
    filter_alg_apertures: list[str] | None = None,
    filter_seeds: list[int] | None = None,
    plot_name: str | None = None,
    ylim: tuple[float, float] | None = None,
    auto_label: bool = False,
    window: int = 1000,
    sort_seeds: bool = False,
):
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
        / "data.parquet"
    )

    # Load processed data
    all_df = pl.read_parquet(data_path)

    # Filter by sample_types
    if sample_types:
        print(list(all_df["sample_type"].unique()))
        all_df = all_df.filter(pl.col("sample_type").is_in(sample_types))

    # Filter algorithms and apertures jointly if specified
    if filter_alg_apertures:
        conditions = []
        for pair in filter_alg_apertures:
            if ":" in pair:
                alg, aperture_str = pair.split(":", 1)
                # Try to convert aperture to int if possible
                try:
                    aperture = int(aperture_str)
                except ValueError:
                    aperture = aperture_str
                conditions.append(
                    (pl.col("alg") == alg) & (pl.col("aperture") == aperture)
                )
            else:
                # If no :, just filter on alg
                conditions.append(pl.col("alg") == pair)
        if conditions:
            combined_condition = conditions[0]
            for cond in conditions[1:]:
                combined_condition = combined_condition | cond
            all_df = all_df.filter(combined_condition)

    # Filter seeds if specified
    if filter_seeds:
        all_df = all_df.filter(pl.col("seed").is_in(filter_seeds))

    # Derive additional columns
    all_df = all_df.with_columns(
        pl.col("alg").str.replace(r"_frozen_.*", "").alias("alg_base"),
        pl.when(pl.col("alg").str.contains("_frozen"))
        .then(pl.col("alg").str.extract(r"_frozen_(.*)", 1))
        .otherwise(None)
        .alias("freeze_steps_str"),
    )

    # Compute metadata from df
    main_alg_apertures = sorted(all_df.select(["alg", "aperture"]).unique().rows())
    if sample_types:
        sample_types_list = [
            st for st in sample_types if st in all_df["sample_type"].unique()
        ]
    else:
        sample_types_list = sorted(all_df["sample_type"].unique())
    print("Main alg-apertures:", main_alg_apertures)
    print("Sample types:", sample_types_list)
    env = all_df["env"][0]

    # Determine biome mappings based on environment
    if "TwoBiome" in env:
        biome_mapping = {-1: "Neither", 0: "Morel", 1: "Oyster"}
    elif "Weather" in env:
        biome_mapping = {-1: "Neither", 0: "Hot", 1: "Cold"}
    else:
        # Fallback - use generic names
        biome_mapping = {}
        if "biome_id" in all_df.columns:
            unique_biomes = sorted(all_df["biome_id"].unique())
            biome_mapping = {biome: f"Biome {biome}" for biome in unique_biomes}

    # Get available biome metrics from data
    available_biomes = []
    if "biome_id" in all_df.columns:
        unique_biomes = sorted(all_df["biome_id"].unique())
        available_biomes = unique_biomes

    # Reorder biomes: morel (0), neither (-1), oyster (1) for TwoBiome; hot (0), neither (-1), cold (1) for Weather
    if "TwoBiome" in env and set(available_biomes) == {-1, 0, 1}:
        available_biomes = [0, -1, 1]
    elif "Weather" in env and set(available_biomes) == {-1, 0, 1}:
        available_biomes = [0, -1, 1]

    biome_metrics = [f"biome_{biome}_occupancy_{window}" for biome in available_biomes]
    biome_names = [
        biome_mapping.get(biome, f"Biome {biome}") for biome in available_biomes
    ]
    # Use specific biome colors: red for morel, blue for neither, yellow for oyster; or for weather: hot, neither, cold
    if "TwoBiome" in env and set(available_biomes) == {-1, 0, 1}:
        # available_biomes is now [0, -1, 1] -> [morel, neither, oyster]
        biome_colors = [
            TWO_BIOME_COLORS["Morel"],
            TWO_BIOME_COLORS["Neither"],
            TWO_BIOME_COLORS["Oyster"],
        ]
        metric_colors = dict(zip(biome_metrics, biome_colors, strict=True))
    elif "Weather" in env and set(available_biomes) == {-1, 0, 1}:
        # available_biomes is now [0, -1, 1] -> [hot, neither, cold]
        biome_colors = [
            WEATHER_BIOME_COLORS["Hot"],
            WEATHER_BIOME_COLORS["Neither"],
            WEATHER_BIOME_COLORS["Cold"],
        ]
        metric_colors = dict(zip(biome_metrics, biome_colors, strict=True))
    else:
        metric_colors = dict(
            zip(biome_metrics, select_colors(len(biome_metrics)), strict=True)
        )

    dd = data_definition(
        hyper_cols=[],
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    # Create figure for stacked bars: rows = algorithms, cols = sample_types
    nrows = len(main_alg_apertures)
    ncols = len(sample_types_list)
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=False, layout="constrained", squeeze=False
    )

    aggregated_data = {}  # ((alg, aperture), sample_type) -> seed -> {metric: value}

    for group_key, df in all_df.group_by(["alg", "aperture", "sample_type"]):
        alg, aperture, sample_type_val = group_key
        print(f"Processing: {alg} - {aperture} - {sample_type_val}")

        key = ((alg, aperture), sample_type_val)
        aggregated_data[key] = {}

        for metric in biome_metrics:
            metric_df = (
                df.sort(dd.seed_col).group_by(dd.seed_col).agg(dd.time_col, metric)
            )
            if metric_df.is_empty() or metric_df[metric].is_null().all():
                continue

            try:
                xs = np.stack(metric_df["frame"].to_list()).astype(np.int64)
            except Exception as e:
                print(f"Skipping {alg} for metric {metric} due to error: {e}")
                continue

            # Handle null values in the metric data
            metric_lists = metric_df[metric].to_list()
            # Check if any list contains null values
            has_nulls = any(
                any(
                    pl.Series(lst).is_null().any()
                    for lst in metric_lists
                    if lst is not None
                )
                for lst in metric_lists
            )

            if has_nulls:
                print(f"Warning: {metric} contains null values, filling with 0")
                # Fill nulls in each list
                cleaned_lists = []
                for lst in metric_lists:
                    if lst is not None:
                        series = pl.Series(lst)
                        filled = series.fill_null(0)
                        cleaned_lists.append(filled.to_list())
                    else:
                        cleaned_lists.append([])
                ys = np.stack(cleaned_lists).astype(np.float64)
            else:
                ys = np.stack(metric_lists).astype(np.float64)
            mask = xs[0] > 1000
            xs, ys = xs[:, mask], ys[:, mask]

            # Skip if we don't have enough data after filtering
            if ys.shape[1] < 10:
                print(
                    f"Skipping {alg} for metric {metric}: insufficient data after filtering"
                )
                continue

            # Take the last value since it's a rolling mean
            final_values_per_seed = ys[:, -1]
            seeds = metric_df[dd.seed_col].to_list()
            for idx, seed in enumerate(seeds):
                if seed not in aggregated_data[key]:
                    aggregated_data[key][seed] = {}
                aggregated_data[key][seed][metric] = final_values_per_seed[idx]

    # Plot the stacked bars
    for i, (alg, aperture) in enumerate(main_alg_apertures):
        for j, sample_type_val in enumerate(sample_types_list):
            ax: Axes = axs[i, j]
            key = ((alg, aperture), sample_type_val)
            if key in aggregated_data:
                seed_data = aggregated_data[key]
                sorted_seeds = seed_data
                if sort_seeds and ("TwoBiome" in env or "Weather" in env):
                    if "TwoBiome" in env:
                        # Sort by highest to lowest morel, lowest to highest oyster
                        morel_metric = f"biome_0_occupancy_{window}"
                        oyster_metric = f"biome_1_occupancy_{window}"
                        sorted_seeds = sorted(
                            seed_data.keys(),
                            key=lambda s: (
                                -seed_data[s].get(morel_metric, 0.0),  # descending morel
                                seed_data[s].get(oyster_metric, 0.0),  # ascending oyster
                            ),
                        )
                    elif "Weather" in env:
                        # Sort by highest to lowest hot, lowest to highest cold
                        hot_metric = f"biome_0_occupancy_{window}"
                        cold_metric = f"biome_1_occupancy_{window}"
                        sorted_seeds = sorted(
                            seed_data.keys(),
                            key=lambda s: (
                                -seed_data[s].get(hot_metric, 0.0),  # descending hot
                                seed_data[s].get(cold_metric, 0.0),  # ascending cold
                            ),
                        )
                for seed_idx, seed in enumerate(sorted_seeds):
                    metrics = seed_data[seed]
                    values = [metrics.get(metric, 0.0) for metric in biome_metrics]
                    colors = [metric_colors[metric] for metric in biome_metrics]
                    left = 0
                    for value, color in zip(values, colors, strict=True):
                        ax.barh(
                            seed_idx,
                            value,
                            left=left,
                            color=color,
                            height=1.0,
                            edgecolor=color,
                        )
                        left += value
                # Remove y ticks
                ax.set_yticks([])
                ax.set_ylim(-0.5, len(sorted_seeds) - 0.5)
                ax.invert_yaxis()  # Flip so seed 0 is on top
            else:
                # No data, set empty ticks
                ax.set_yticks([])
                ax.set_ylim(-0.5, -0.5)

    # Set row labels (algorithms)
    for i, (alg, aperture) in enumerate(main_alg_apertures):
        alg_label = str(LABEL_MAP.get(alg, alg))
        if aperture is not None:
            alg_label += f" FOV {aperture}"
        axs[i, 0].set_ylabel(alg_label, rotation=0, ha="right", va="center")

    # Set column labels (sample types)
    for j, sample_type_val in enumerate(sample_types_list):
        label = SAMPLE_TYPE_MAP.get(sample_type_val, sample_type_val)
        axs[-1, j].set_xlabel(label, labelpad=10)

    # Remove ax lines (spines)
    for ax in axs.flatten():
        ax.set_xticks([])  # Hide x ticks
        ax.set_xticklabels([])  # Hide x tick labels
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Set xlim if ylim provided (note: ylim param sets xlim for horizontal bars)
    if ylim is not None:
        xlim = ylim
    else:
        xlim = (0.0, 1.0)
    for ax in axs.flatten():
        ax.set_xlim(xlim)

    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=metric_colors[m], label=n)
        for m, n in zip(biome_metrics, biome_names, strict=True)
    ]
    fig.legend(
        handles=legend_elements,
        loc="outside upper center",
        frameon=False,
        ncol=len(legend_elements),
    )

    base_name = plot_name if plot_name else env

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=f"{base_name}_biome_occupancy_bars",
        save_type=save_type,
        f=fig,
        width=ncols if ncols > 1 else (1 if auto_label else 2),
        height_ratio=(1 / ncols) * (2 / 3)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot biome occupancy metrics from processed data"
    )
    parser.add_argument("path", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--save-type",
        type=str,
        default="pdf",
        help="File format to save the plots (default: pdf)",
    )
    parser.add_argument(
        "--sample-types",
        nargs="*",
        default=["every"],
        help="Sample types to plot (default: every)",
    )
    parser.add_argument(
        "--filter-alg-apertures",
        nargs="*",
        default=None,
        help="List of alg:aperture pairs to include (e.g., DQN:9 DQN:15), or just alg for any aperture (default: all)",
    )
    parser.add_argument(
        "--filter-seeds",
        nargs="*",
        type=int,
        default=None,
        help="List of seed numbers to include (default: all)",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=None,
        help="Override the default plot name (default: use environment name)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis limits as two floats (default: auto)",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Use automatic label annotation instead of legend",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Window size for biome occupancy metric (default: 1000)",
    )
    parser.add_argument(
        "--sort-seeds",
        action="store_true",
        help="Sort seeds by highest to lowest morel, lowest to highest oyster (TwoBiome) or highest to lowest hot, lowest to highest cold (Weather)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.save_type,
        args.sample_types,
        args.filter_alg_apertures,
        args.filter_seeds,
        args.plot_name,
        args.ylim,
        args.auto_label,
        args.window,
        args.sort_seeds,
    )

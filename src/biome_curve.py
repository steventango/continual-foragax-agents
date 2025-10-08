import argparse
import json
import math
import os
import sys
from pathlib import Path

from annotate_plot import annotate_plot

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from plotting_utils import (
    TWO_BIOME_COLORS,
    WEATHER_BIOME_COLORS,
    PlottingArgumentParser,
    despine,
    load_data,
    parse_plotting_args,
    save_plot,
)


def main(
    experiment_path: Path,
    save_type: str = "pdf",
    sample_type: str = "every",
    filter_algs: list[str] | None = None,
    plot_name: str | None = None,
    ylim: tuple[float, float] | None = None,
    auto_label: bool = False,
    window: int = 1000,
):
    parser = PlottingArgumentParser(description="Plot biome occupancy curves.")
    parser.add_argument(
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to filter from the data.",
    )
    parser.add_argument(
        "--window", type=int, default=1000, help="Occupancy window size."
    )
    parser.add_argument("--ylim", type=float, nargs=2, help="Y-axis limits.")
    parser.add_argument(
        "--auto-label", action="store_true", help="Enable auto-labeling."
    )

    args = parse_plotting_args(parser)

    # Load processed data
    df = load_data(args.experiment_path)
    df = df.filter(pl.col("sample_type") == args.sample_type)

    # Filter algorithms if specified
    if args.filter_algs:
        df = df.filter(pl.col("alg").is_in(args.filter_algs))

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    env = df["env"][0]

    palette = None
    # Determine biome mappings and color palette based on environment
    if "TwoBiome" in env:
        biome_mapping = {-1: "Neither", 0: "Morel", 1: "Oyster"}
        palette = TWO_BIOME_COLORS
    elif "Weather" in env:
        biome_mapping = {-1: "Neither", 0: "Hot", 1: "Cold"}
        palette = WEATHER_BIOME_COLORS
    else:
        # Fallback - use generic names
        biome_mapping = {
            b: f"Biome {b}" for b in df["biome_id"].unique() if b is not None
        }

    available_biomes = sorted(biome_mapping.keys())
    biome_metrics = [f"biome_{b}_occupancy_{args.window}" for b in available_biomes]
    biome_names = [biome_mapping[b] for b in available_biomes]

    df_melted = df.melt(
        id_vars=["frame", "alg", "seed"],
        value_vars=biome_metrics,
        variable_name="metric",
        value_name="value",
    )

    metric_to_name = dict(zip(biome_metrics, biome_names, strict=True))
    df_melted = df_melted.with_columns(
        pl.col("metric").replace(metric_to_name).alias("Biome")
    )

    g = sns.relplot(
        data=df_melted.to_pandas(),
        x="frame",
        y="value",
        hue="Biome",
        col="alg",
        kind="line",
        errorbar=("ci", 95),
        col_wrap=min(len(df["alg"].unique()), 3),
        facet_kws=dict(sharey=True),
        palette=palette,
    )

    g.set_axis_labels("Time steps", "Occupancy")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle(f"{env} - Biome Occupancy", y=1.03)

    if args.ylim:
        g.set(ylim=args.ylim)

    for ax in g.axes.flatten():
        despine(ax)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        if args.auto_label:
            annotate_plot(ax)

    if not args.auto_label:
        g.add_legend(title=None, frameon=False)

    plot_name = args.plot_name if args.plot_name else f"{env}_biome_occupancy"
    save_plot(g.fig, args.experiment_path, plot_name, args.save_type)

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
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to plot (default: every)",
    )
    parser.add_argument(
        "--filter-algs",
        nargs="*",
        default=None,
        help="List of algorithms to show (default: all)",
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
    args = parser.parse_args()

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.save_type,
        args.sample_type,
        args.filter_algs,
        args.plot_name,
        args.ylim,
        args.auto_label,
        args.window,
    )

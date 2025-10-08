import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")

import matplotlib.path as mpath
import numpy as np
import polars as pl
import seaborn as sns
import ternary
from matplotlib.patches import Patch

from plotting_utils import (
    LABEL_MAP,
    PlottingArgumentParser,
    load_data,
    parse_plotting_args,
    save_plot,
)

SAMPLE_TYPE_MAP = {
    "999000:1000000:500": "Early learning",
    "4999000:5000000:500": "Mid learning",
    "9999000:10000000:500": "Late learning",
}


def parse_bars_ternary(
    bar_strings: list[str] | None,
) -> list[tuple[str, str | float | None, str, list[int] | None]] | None:
    if not bar_strings:
        return None
    bars = []
    for bar_str in bar_strings:
        parts = bar_str.split("|")
        alg, aperture_str, sample_type, seeds_str = parts
        aperture = float(aperture_str) if aperture_str else None
        seeds = [int(s) for s in seeds_str.split(",")] if seeds_str else None
        bars.append((alg, aperture, sample_type, seeds))
    return bars


def main():
    parser = PlottingArgumentParser(
        description="Plot biome occupancy in a ternary plot."
    )
    parser.add_argument(
        "--bars", nargs="*", help="Bar specifications for ternary plot points."
    )
    parser.add_argument(
        "--window", type=int, default=1000000, help="Occupancy window size."
    )

    args = parse_plotting_args(parser)
    bars = parse_bars_ternary(args.bars)

    df = load_data(args.experiment_path)
    env = df["env"][0]

    if "TwoBiome" in env:
        biome_mapping = {-1: "Neither", 0: "Morel", 1: "Oyster"}
    elif "Weather" in env:
        biome_mapping = {-1: "Neither", 0: "Hot", 1: "Cold"}
    else:
        raise ValueError(f"Unknown biome mapping for environment: {env}")

    available_biomes = sorted(df["biome_id"].unique())
    biome_metrics = [f"biome_{b}_occupancy_{args.window}" for b in available_biomes]
    biome_names = [biome_mapping[b] for b in available_biomes]

    if not bars:
        unique_algs = df["alg"].unique().to_list()
        bars = [(alg, None, "every", None) for alg in unique_algs]

    fig, tax = ternary.figure(scale=100)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=20)
    tax.set_title(f"{env}\n")

    tax.left_corner_label(biome_names[2])
    tax.right_corner_label(biome_names[0])
    tax.top_corner_label(biome_names[1])

    plot_data = []
    for alg, aperture, sample_type, seeds in bars:
        bar_df = df.filter(pl.col("sample_type") == sample_type)
        if aperture is not None:
            bar_df = bar_df.filter(pl.col("aperture") == aperture)
        if seeds is not None:
            bar_df = bar_df.filter(pl.col("seed").is_in(seeds))
        bar_df = bar_df.filter(pl.col("alg") == alg)

        last_frame = bar_df["frame"].max()
        bar_df = bar_df.filter(pl.col("frame") == last_frame)

        points = bar_df.select(biome_metrics).to_numpy() * 100

        label = LABEL_MAP.get(alg, alg)
        for point in points:
            plot_data.append({"point": tuple(point), "label": label})

    plot_df = pl.DataFrame(plot_data)

    # Plotting points
    for label in plot_df["label"].unique():
        points_to_plot = plot_df.filter(pl.col("label") == label)["point"].to_list()
        tax.scatter(points_to_plot, label=label)

    tax.legend()

    plot_name = args.plot_name or f"{env}_biome_ternary"
    save_plot(
        fig, args.experiment_path, plot_name, args.save_type, width=1, height_ratio=1
    )


if __name__ == "__main__":
    main()

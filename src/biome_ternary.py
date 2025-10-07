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
from PyExpPlotting.matplot import save
from ternary.helpers import project_point

from utils.constants import LABEL_MAP
from utils.plotting import select_colors


def main(
    experiment_path: Path,
    save_type: str = "pdf",
    bars: list[tuple[str, str, list[int] | None]] | None = None,
    plot_name: str | None = None,
    window: int = 1000000,
):
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
        / "data.parquet"
    )

    # Load processed data
    all_df = pl.read_parquet(data_path)
    print(list(all_df["sample_type"].unique()))

    # Derive additional columns
    all_df = all_df.with_columns(
        pl.col("alg").str.replace(r"_frozen_.*", "").alias("alg_base"),
        pl.when(pl.col("alg").str.contains("_frozen"))
        .then(pl.col("alg").str.extract(r"_frozen_(.*)", 1))
        .otherwise(None)
        .alias("freeze_steps_str"),
    )

    print("Available alg_base:", sorted(all_df["alg_base"].unique()))

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

    biome_names = [
        biome_mapping.get(biome, f"Biome {biome}") for biome in available_biomes
    ]

    # If no bars specified, use default behavior (all algs with default sample_type and seeds)
    if bars is None:
        # Default: all algs, sample_type="end", all seeds
        unique_algs = all_df["alg"].unique().to_list()
        bars = [(alg, "end", None) for alg in unique_algs]

    # Collect agg_data for each bar
    agg_data = []
    for bar_alg, bar_sample_type, bar_seeds in bars:
        bar_agg_data = []
        # Filter data for this bar
        bar_df = all_df.filter(pl.col("sample_type") == bar_sample_type)
        if bar_seeds is not None:
            bar_df = bar_df.filter(pl.col("seed").is_in(bar_seeds))

        # Filter to the alg
        bar_df = bar_df.filter(pl.col("alg") == bar_alg)

        # Filter data to frame == target_frame (or max frame if less)
        max_frame = bar_df.select(pl.col("frame").max()).item()
        filtered_df = bar_df.filter(pl.col("frame") == max_frame)

        print(
            f"For bar {bar_alg} {bar_sample_type} seeds {bar_seeds}: filtered df shape {filtered_df.shape}"
        )

        # Aggregate biome occupancy for this bar
        for group_key, df in filtered_df.group_by(
            ["alg_base", "aperture", "alg", "seed"]
        ):
            alg_base, aperture, alg, seed = group_key
            for biome in available_biomes:
                col_name = f"biome_{biome}_occupancy_{window}"
                if col_name in df.columns:
                    mean_val = df[col_name].item()
                    if mean_val is not None and mean_val == mean_val:
                        bar_agg_data.append(
                            {
                                "bar_alg": bar_alg,
                                "bar_sample_type": bar_sample_type,
                                "bar_seeds": bar_seeds,
                                "bar_seeds_str": ",".join(map(str, bar_seeds))
                                if bar_seeds
                                else "",
                                "alg_base": alg_base,
                                "aperture": aperture,
                                "alg": alg,
                                "seed": seed,
                                "biome": biome,
                                "biome_name": biome_mapping.get(
                                    biome, f"Biome {biome}"
                                ),
                                "mean_occupancy": mean_val,
                            }
                        )

        if not bar_agg_data:
            raise ValueError(
                f"No data available for bar: {bar_alg}, {bar_sample_type}, seeds: {bar_seeds}"
            )

        agg_data.extend(bar_agg_data)

    # Prepare data for ternary plotting
    bar_points = [[] for _ in bars]
    bar_labels = []
    bar_colors = select_colors(len(bars))

    for bar_idx, (bar_alg, bar_sample_type, bar_seeds) in enumerate(bars):
        # Filter data for this bar
        bar_df = all_df.filter(pl.col("sample_type") == bar_sample_type)
        if bar_seeds is not None:
            bar_df = bar_df.filter(pl.col("seed").is_in(bar_seeds))

        # Filter to the alg
        bar_df = bar_df.filter(pl.col("alg") == bar_alg)

        # Filter data to frame == target_frame (or max frame if less)
        max_frame = bar_df.select(pl.col("frame").max()).item()
        filtered_df = bar_df.filter(pl.col("frame") == max_frame)

        # Get unique seeds for this bar
        unique_seeds = sorted(filtered_df["seed"].unique())

        # Create label for this bar
        bar_label = str(LABEL_MAP.get(bar_alg, bar_alg))
        if bar_sample_type != "end":
            bar_label += f" {bar_sample_type}"
        bar_labels.append(bar_label)

        for seed in unique_seeds:
            seed_df = filtered_df.filter(pl.col("seed") == seed)

            # Get the three proportions for this seed
            proportions = []
            for biome in available_biomes:
                col_name = f"biome_{biome}_occupancy_{window}"
                if col_name in seed_df.columns:
                    mean_val = seed_df[col_name].item()
                    if mean_val is not None and not (
                        mean_val != mean_val
                    ):  # Check for NaN
                        proportions.append(mean_val)
                    else:
                        raise ValueError(
                            f"Missing data for biome {biome} in bar {bar_alg}, sample_type {bar_sample_type}, seed {seed}"
                        )
                else:
                    raise ValueError(
                        f"Missing column {col_name} for biome {biome} in bar {bar_alg}, sample_type {bar_sample_type}, seed {seed}"
                    )

            # Normalize to sum to 100 (ternary scale)
            total = sum(proportions)
            if total > 0:
                proportions = [p / total * 100 for p in proportions]
            else:
                proportions = [0, 0, 0]

            print(
                f"Bar {bar_alg} {bar_sample_type} seed {seed}: proportions {proportions}"
            )

            bar_points[bar_idx].append(proportions)  # Plot ternary diagram
    fig, tax = ternary.figure(scale=100)
    tax.set_background_color("white")  # Make background white
    tax.boundary(linewidth=0.0)  # Remove the box around the ternary plot
    tax.gridlines(multiple=20, color="gray", linewidth=0.5)

    # Hide matplotlib axes border
    tax.ax.spines["top"].set_visible(False)
    tax.ax.spines["right"].set_visible(False)
    tax.ax.spines["bottom"].set_visible(False)
    tax.ax.spines["left"].set_visible(False)

    # Hide matplotlib ticks and labels
    tax.clear_matplotlib_ticks()

    # Use ternary ticks and labels
    tax.ticks(axis="lbr", multiple=20, linewidth=0, tick_formats="%.0f%%", offset=0.03)

    # Set axis labels
    tax.set_title(f"{env} - Biome Occupancy", fontsize=16, pad=20)
    tax.left_axis_label(biome_names[2], fontsize=12, offset=0.2)
    tax.right_axis_label(biome_names[0], fontsize=12, offset=0.2)
    tax.bottom_axis_label(biome_names[1], fontsize=12, offset=0.2)
    # Prepare combined data for hue-based KDE plotting
    all_x = []
    all_y = []
    all_hue = []
    hue_labels = []
    hue_colors = []

    for bar_idx, points_list in enumerate(bar_points):
        if points_list:
            cartesian = np.array([project_point(p) for p in points_list])
            x_coords = cartesian[:, 0]
            y_coords = cartesian[:, 1]

            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_hue.extend([bar_idx] * len(points_list))

            hue_labels.append(bar_labels[bar_idx])
            hue_colors.append(bar_colors[bar_idx])

    # Convert to numpy arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_hue = np.array(all_hue)

    # Create color palette for hue
    hue_palette = {i: color for i, color in enumerate(hue_colors)}

    # Plot all KDEs at once using hue
    triangle_vertices = [(0, 0), (100, 0), (50, 86.60254037844386)]
    sns.kdeplot(
        x=all_x,
        y=all_y,
        hue=all_hue,
        ax=tax.ax,
        palette=hue_palette,
        alpha=0.7,
        bw_adjust=0.4,
        levels=10,
        clip=(0, 100),
        fill=True,
    )
    # Clip all KDE collections to the triangle
    path = mpath.Path(triangle_vertices)
    for collection in tax.ax.collections[-len(bar_points):]:  # Clip the last N collections (one per hue level)
        collection.set_clip_path(path, transform=tax.ax.transData)

    # for bar_idx, points_list in enumerate(bar_points):
    #     if points_list:
    #         tax.scatter(
    #             points_list, marker="o", color=bar_colors[bar_idx], s=10, alpha=0.3
    #         )

    # Create legend manually since seaborn hue doesn't always auto-create it
    legend_elements = []
    for hue_idx, bar_label in enumerate(hue_labels):
        legend_elements.append(
            Patch(
                facecolor=hue_colors[hue_idx],
                label=bar_label,
                alpha=1,
            )
        )
    tax.ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=10,
    )

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=plot_name or f"{env}_biome_occupancy_ternary",
        save_type=save_type,
        f=fig,  # type: ignore
        width=1,
        height_ratio=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot biome occupancy bar plots from processed data"
    )
    parser.add_argument("path", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--save-type",
        type=str,
        default="pdf",
        help="File format to save the plots (default: pdf)",
    )
    parser.add_argument(
        "--bars",
        nargs="*",
        help="Bar specifications in format 'alg:sample_type:seeds' where seeds is comma-separated (e.g., 'DQN:slice_1000000_1000_500:0,1')",
    )
    parser.add_argument(
        "--filter-algs",
        nargs="*",
        help="Filter algorithms to plot (by alg_base) - for backward compatibility",
    )
    parser.add_argument(
        "--filter-seeds",
        nargs="*",
        type=int,
        help="Filter seeds to plot - for backward compatibility",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="end",
        help="Sample type to plot (default: end) - for backward compatibility",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=None,
        help="Custom plot name (default: {env}_biome_occupancy_bar)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Biome occupancy window size in steps (default: 1000)",
    )
    args = parser.parse_args()

    # Parse bars
    bars = None
    if args.bars:
        bars = []
        for bar_spec in args.bars:
            parts = bar_spec.split("|")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid bar spec: {bar_spec}. Expected format 'alg|sample_type|seeds'"
                )
            alg, sample_type, seeds_str = parts
            if seeds_str:
                seeds = [int(s.strip()) for s in seeds_str.split(",")]
            else:
                seeds = None
            bars.append((alg, sample_type, seeds))
    elif args.filter_algs or args.filter_seeds or args.sample_type != "end":
        # Backward compatibility: construct bars from old args
        if not args.filter_algs:
            # If no filter_algs, use all
            # But since we don't have all_df yet, we'll handle in main
            pass
        else:
            bars = []
            for alg in args.filter_algs:
                bars.append((alg, args.sample_type, args.filter_seeds))

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.save_type,
        bars,
        args.plot_name,
        args.window,
    )

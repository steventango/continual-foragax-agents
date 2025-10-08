import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
from PyExpPlotting.matplot import save

from utils.constants import LABEL_MAP
from utils.plotting import select_colors

SAMPLE_TYPE_MAP = {
    "999000:1000000:500": "Early learning",
    "4999000:5000000:500": "Mid learning",
    "9999000:10000000:500": "Late learning",
}


def main(
    experiment_path: Path,
    save_type: str = "pdf",
    bars: list[tuple[str, str | float | None, str, list[int] | None]] | None = None,
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
    print("Available alg:", sorted(all_df["alg"].unique()))

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

    # Determine biome mappings based on environment (only Morel and Oyster)
    if "TwoBiome" in env:
        biome_mapping = {0: "Morel", 1: "Oyster"}
    elif "Weather" in env:
        biome_mapping = {0: "Hot", 1: "Cold"}
    else:
        # Fallback - use generic names for biomes 0 and 1
        biome_mapping = {0: "Biome 0", 1: "Biome 1"}

    # Get available biome metrics from data (only 0 and 1)
    available_biomes = [0, 1]  # Only Morel and Oyster

    biome_names = [
        biome_mapping.get(biome, f"Biome {biome}") for biome in available_biomes
    ]

    # If no bars specified, use default behavior (all algs with default sample_type and seeds)
    if bars is None:
        # Default: all algs, sample_type="end", all seeds
        unique_algs = all_df["alg"].unique().to_list()
        bars = [(alg, None, "end", None) for alg in unique_algs]

    # Collect data for plotting
    bar_points = [[] for _ in bars]
    bar_labels = []
    bar_colors = select_colors(len(bars), override="vibrant")

    for bar_idx, (bar_alg, bar_aperture, bar_sample_type, bar_seeds) in enumerate(bars):
        # Filter data for this bar
        bar_df = all_df.filter(pl.col("sample_type") == bar_sample_type)
        if bar_seeds is not None:
            bar_df = bar_df.filter(pl.col("seed").is_in(bar_seeds))

        # Filter to the alg
        bar_df = bar_df.filter(pl.col("alg") == bar_alg)
        if bar_aperture is not None:
            bar_df = bar_df.filter(pl.col("aperture") == bar_aperture)

        print(
            f"For plotting bar {bar_alg}: alg values in bar_df: {sorted(bar_df['alg'].unique())}"
        )

        # Get unique seeds for this bar
        unique_seeds = sorted(bar_df["seed"].unique())

        # Create label for this bar
        bar_label = str(LABEL_MAP.get(bar_alg, bar_alg))
        if bar_aperture is not None:
            bar_label += f" FOV {bar_aperture}"
        if bar_sample_type != "end":
            readable_sample = SAMPLE_TYPE_MAP.get(bar_sample_type, bar_sample_type)
            bar_label += f" {readable_sample}"
        bar_labels.append(bar_label)

        for seed in unique_seeds:
            seed_df = bar_df.filter(pl.col("seed") == seed)

            # Get proportions for Morel (0) and Oyster (1)
            morel_occupancy = None
            oyster_occupancy = None

            for biome in available_biomes:
                col_name = f"biome_{biome}_occupancy_{window}"
                if col_name in seed_df.columns:
                    # Take the last value since it's a rolling mean
                    max_frame = seed_df.select(pl.col("frame").max()).item()
                    mean_val = seed_df.filter(pl.col("frame") == max_frame)[
                        col_name
                    ].item()
                    if mean_val is not None and not (
                        mean_val != mean_val
                    ):  # Check for NaN
                        if biome == 0:
                            morel_occupancy = mean_val
                        elif biome == 1:
                            oyster_occupancy = mean_val
                    else:
                        raise ValueError(
                            f"Missing data for biome {biome} in bar {bar_alg}, aperture {bar_aperture}, sample_type {bar_sample_type}, seed {seed}"
                        )
                else:
                    raise ValueError(
                        f"Missing column {col_name} for biome {biome} in bar {bar_alg}, aperture {bar_aperture}, sample_type {bar_sample_type}, seed {seed}"
                    )

            if morel_occupancy is not None and oyster_occupancy is not None:
                morel_norm = morel_occupancy * 100
                oyster_norm = oyster_occupancy * 100

                bar_points[bar_idx].append((morel_norm, oyster_norm))

                print(
                    f"Bar {bar_alg} aperture {bar_aperture} {bar_sample_type} seed {seed}: Morel {morel_norm:.2f}%, Oyster {oyster_norm:.2f}%"
                )
            else:
                raise ValueError(
                    f"Missing occupancy data for bar {bar_alg}, aperture {bar_aperture}, sample_type {bar_sample_type}, seed {seed}"
                )

    # Create 2D scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"{biome_names[0]} Occupancy (%)", fontsize=12)
    ax.set_ylabel(f"{biome_names[1]} Occupancy (%)", fontsize=12)
    ax.set_title(f"{env} - {biome_names[0]} vs {biome_names[1]} Occupancy", fontsize=16, pad=20)
    ax.grid(False)
    ax.set_aspect('equal')

    # Plot points for each bar
    for bar_idx, points_list in enumerate(bar_points):
        if points_list:
            x_coords = [p[0] for p in points_list]
            y_coords = [p[1] for p in points_list]
            ax.scatter(
                x_coords,
                y_coords,
                color=bar_colors[bar_idx],
                s=8,
                alpha=0.7,
                linewidth=0,
                label=bar_labels[bar_idx],
            )

    # Add legend
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=10,
    )

    # Adjust layout to make room for legend
    plt.tight_layout()

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=plot_name or f"{env}_biome_occupancy_scatter",
        save_type=save_type,
        f=fig,  # type: ignore
        width=1,
        height_ratio=1/2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot biome occupancy scatter plot (Morel vs Oyster) from processed data"
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
        help="Bar specifications in format 'alg:aperture|sample_type|seeds' or 'alg|sample_type|seeds' where seeds is comma-separated (e.g., 'DQN:9|end|' or 'DQN|end|0,1')",
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
        help="Custom plot name (default: {env}_biome_occupancy_scatter)",
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
            if len(parts) == 3:
                # Could be old format or new with aperture in first part
                first_part = parts[0]
                if ":" in first_part:
                    # New format: alg:aperture|sample_type|seeds
                    alg_aperture, sample_type, seeds_str = parts
                    if ":" in alg_aperture:
                        alg, aperture_str = alg_aperture.split(":", 1)
                        try:
                            aperture = int(aperture_str)
                        except ValueError:
                            aperture = aperture_str
                    else:
                        alg = alg_aperture
                        aperture = None
                else:
                    # Old format: alg|sample_type|seeds
                    alg, sample_type, seeds_str = parts
                    aperture = None
            elif len(parts) == 4:
                # New format: alg|aperture|sample_type|seeds
                alg, aperture_str, sample_type, seeds_str = parts
                if aperture_str.strip():
                    try:
                        aperture = int(aperture_str)
                    except ValueError:
                        aperture = aperture_str
                else:
                    aperture = None
            else:
                raise ValueError(
                    f"Invalid bar spec: {bar_spec}. Expected 'alg:aperture|sample_type|seeds' or 'alg|sample_type|seeds'"
                )
            if seeds_str:
                seeds = [int(s.strip()) for s in seeds_str.split(",")]
            else:
                seeds = None
            bars.append((alg, aperture, sample_type, seeds))
    elif args.filter_algs or args.filter_seeds or args.sample_type != "end":
        # Backward compatibility: construct bars from old args
        if not args.filter_algs:
            # If no filter_algs, use all
            # But since we don't have all_df yet, we'll handle in main
            pass
        else:
            bars = []
            for alg in args.filter_algs:
                bars.append((alg, None, args.sample_type, args.filter_seeds))

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.save_type,
        bars,
        args.plot_name,
        args.window,
    )

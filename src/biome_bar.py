import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from scipy.stats import bootstrap

from utils.constants import LABEL_MAP
from utils.plotting import select_colors

setDefaultConference("jmlr")
setFonts(20)

def format_sample_type(sample_type: str) -> str:
    """Format sample_type for display, e.g., 'slice_1000000_1000_500' -> '[1000000:1001000:500]'"""
    if sample_type.startswith("slice_"):
        parts = sample_type.split("_")
        if len(parts) == 4:
            _, start_str, length_str, stride_str = parts
            try:
                start = int(start_str)
                length = int(length_str)
                stride = int(stride_str)
                end = start + length
                return f"[{start}:{end}:{stride}]"
            except ValueError:
                pass
    return sample_type


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
    metric_colors = dict(
        zip([f"biome_{biome}" for biome in available_biomes], select_colors(len(available_biomes)), strict=True)
    )

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

    agg_df = pl.DataFrame(agg_data)

    # Prepare data for plotting with bootstrapped confidence intervals
    bar_labels = []
    biome_data = {biome: [] for biome in biome_names}
    biome_errors_lower = {biome: [] for biome in biome_names}
    biome_errors_upper = {biome: [] for biome in biome_names}

    for bar_alg, bar_sample_type, bar_seeds in bars:
        # Generate label for this bar
        alg_label = str(LABEL_MAP.get(bar_alg, bar_alg))
        if bar_sample_type != "end":
            formatted_sample = format_sample_type(bar_sample_type)
            alg_label += f"\n{formatted_sample}"
        if bar_seeds is not None and len(bar_seeds) == 1:
            alg_label += f"\n(seed {bar_seeds[0]})"
        elif bar_seeds is not None:
            alg_label += f"\n(seeds {','.join(map(str, bar_seeds))})"
        bar_labels.append(alg_label)

        for biome_name in biome_names:
            # Get all seed values for this bar/biome combination
            bar_seeds_str = ",".join(map(str, bar_seeds)) if bar_seeds else ""
            seed_values = agg_df.filter(
                (pl.col("bar_alg") == bar_alg)
                & (pl.col("bar_sample_type") == bar_sample_type)
                & (pl.col("bar_seeds_str") == bar_seeds_str)
                & (pl.col("biome_name") == biome_name)
            )["mean_occupancy"].to_numpy()

            if len(seed_values) == 0:
                mean_val = 0
                lower_err = 0
                upper_err = 0
            elif len(seed_values) == 1:
                mean_val = seed_values[0]
                lower_err = np.nan
                upper_err = np.nan
            else:
                mean_val = np.mean(seed_values)
                # Use scipy.stats.bootstrap for 95% confidence interval
                boot_result = bootstrap((seed_values,), np.mean, confidence_level=0.95, n_resamples=1000)
                ci_lower = boot_result.confidence_interval.low
                ci_upper = boot_result.confidence_interval.high
                lower_err = mean_val - ci_lower
                upper_err = ci_upper - mean_val

            biome_data[biome_name].append(mean_val)
            biome_errors_lower[biome_name].append(lower_err)
            biome_errors_upper[biome_name].append(upper_err)

    # Plot bar chart
    fig, ax = plt.subplots(layout="constrained")

    x = np.arange(len(bar_labels))
    width = 0.8 / len(biome_names)  # Width of each bar

    for i, biome_name in enumerate(biome_names):
        means = biome_data[biome_name]
        lower_errors = biome_errors_lower[biome_name]
        upper_errors = biome_errors_upper[biome_name]
        color = metric_colors[f"biome_{available_biomes[i]}"]
        ax.bar(
            x + i * width - width * (len(biome_names) - 1) / 2,
            means,
            width,
            label=biome_name,
            color=color,
            yerr=[lower_errors, upper_errors],
            capsize=3
        )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Proportion of Time in Biome")
    ax.set_title(f"{env} - Biome Occupancy")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    save(
        save_path=f"{experiment_path}/plots",
        plot_name=plot_name or f"{env}_biome_occupancy_bar",
        save_type=save_type,
        f=fig,
        width=len(bar_labels),
        height_ratio=2/len(bar_labels),
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
            parts = bar_spec.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid bar spec: {bar_spec}. Expected format 'alg:sample_type:seeds'"
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

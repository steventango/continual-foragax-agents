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


def main(
    experiment_path: Path,
    save_type: str = "pdf",
    filter_algs: list[str] | None = None,
    plot_name: str | None = None,
    sample_type: str = "every",
    filter_seeds: list[int] | None = None,
):
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
        / "data.parquet"
    )

    # Load processed data
    all_df = pl.read_parquet(data_path)
    print(list(all_df["sample_type"].unique()))
    # Filter by sample_type
    all_df = all_df.filter(pl.col("sample_type") == sample_type)

    # Filter by seeds if specified
    if filter_seeds:
        print(f"Filtering to seeds: {filter_seeds}")
        all_df = all_df.filter(pl.col("seed").is_in(filter_seeds))

    # Derive additional columns
    all_df = all_df.with_columns(
        pl.col("alg").str.replace(r"_frozen_.*", "").alias("alg_base"),
        pl.when(pl.col("alg").str.contains("_frozen"))
        .then(pl.col("alg").str.extract(r"_frozen_(.*)", 1))
        .otherwise(None)
        .alias("freeze_steps_str"),
    )

    print("Available alg_base:", sorted(all_df["alg_base"].unique()))

    # Filter algorithms if specified
    if filter_algs:
        print(f"Filtering to algorithms: {filter_algs}")
        all_df = all_df.filter(pl.col("alg_base").is_in(filter_algs))

    # Compute metadata from df
    main_algs = sorted(
        all_df["alg_base", "aperture"].unique().iter_rows(named=False)
    )
    print("Main algs:", main_algs)
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

    # Filter data to frame == target_frame (or max frame if less)
    max_frame = all_df.select(pl.col("frame").max()).item()
    filtered_df = all_df.filter(pl.col("frame") == max_frame)

    print(
        f"After filtering to frame {filtered_df['frame'].unique()[0] if len(filtered_df) > 0 else 'N/A'}, unique alg_base: {sorted(filtered_df['alg_base'].unique())}"
    )
    print(f"Filtered df shape: {filtered_df.shape}")

    # Use pre-calculated biome_occupancy_{i}_1000000 metric
    agg_data = []
    for group_key, df in filtered_df.group_by(["alg_base", "aperture", "alg", "seed"]):
        alg_base, aperture, alg, seed = group_key
        for biome in available_biomes:
            col_name = f"biome_{biome}_occupancy_1000000"
            if col_name in df.columns:
                mean_val = df[col_name].item()
                if mean_val is not None and mean_val == mean_val:
                    agg_data.append({
                        "alg_base": alg_base,
                        "aperture": aperture,
                        "alg": alg,
                        "seed": seed,
                        "biome": biome,
                        "biome_name": biome_mapping.get(biome, f"Biome {biome}"),
                        "mean_occupancy": mean_val
                    })

    agg_df = pl.DataFrame(agg_data)

    # Prepare data for plotting with bootstrapped confidence intervals
    alg_labels = []
    biome_data = {biome: [] for biome in biome_names}
    biome_errors_lower = {biome: [] for biome in biome_names}
    biome_errors_upper = {biome: [] for biome in biome_names}

    for alg_key in main_algs:
        alg_base, aperture = alg_key
        alg_label = str(LABEL_MAP.get(alg_base, alg_base))
        if aperture is not None:
            alg_label += f" ({aperture})"
        alg_labels.append(alg_label)

        for biome_name in biome_names:
            # Get all seed values for this alg/biome combination
            seed_values = agg_df.filter(
                (pl.col("alg_base") == alg_base) &
                (pl.col("aperture").is_null() if aperture is None else pl.col("aperture") == aperture) &
                (pl.col("biome_name") == biome_name)
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

    # Wrap long labels with line breaks
    def wrap_label(label, max_length=15):
        if len(label) <= max_length:
            return label
        # Prioritize breaking at " ("
        if " (" in label:
            paren_idx = label.find(" (")
            if paren_idx > 0:  # Make sure there's something before the parenthesis
                return '\n'.join([label[:paren_idx], label[paren_idx+1:]])  # +1 to skip the space
        # Try to break at spaces
        words = label.split()
        if len(words) > 1:
            mid = len(words) // 2
            return '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
        # If no spaces, break at max_length
        return '\n'.join([label[:max_length], label[max_length:]])

    alg_labels = [wrap_label(label) for label in alg_labels]

    # Plot bar chart
    fig, ax = plt.subplots(layout="constrained")

    x = np.arange(len(alg_labels))
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

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Proportion of Time in Biome")
    ax.set_title(f"{env} - Biome Occupancy")
    ax.set_xticks(x)
    ax.set_xticklabels(alg_labels)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=plot_name or f"{env}_biome_occupancy_bar",
        save_type=save_type,
        f=fig,
        width=1,
        height_ratio=4/3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot biome occupancy bar plots from processed data (last 1M steps)"
    )
    parser.add_argument("path", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--save-type",
        type=str,
        default="pdf",
        help="File format to save the plots (default: pdf)",
    )
    parser.add_argument(
        "--filter-algs",
        nargs="*",
        help="Filter algorithms to plot (by alg_base)",
    )
    parser.add_argument(
        "--filter-seeds",
        nargs="*",
        type=int,
        help="Filter seeds to plot",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=None,
        help="Custom plot name (default: {env}_biome_occupancy_bar)",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="end",
        help="Sample type to plot (default: end)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.save_type,
        args.filter_algs,
        args.plot_name,
        args.sample_type,
        args.filter_seeds,
    )

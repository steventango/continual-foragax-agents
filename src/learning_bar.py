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

def format_metric_name(metric: str) -> str:
    """Format metric name for display (e.g., 'mean_reward' -> 'Mean Reward')."""
    return metric.replace('_', ' ').title()

def main(
    experiment_path: Path,
    save_type: str = "pdf",
    bars: list[tuple[str, int | None, str, list[int] | None]] | None = None,
    plot_name: str | None = None,
    metric: str = "mean_reward",
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

    # If no bars specified, use default behavior (all algs with default sample_type and seeds)
    if bars is None:
        # Default: all algs, sample_type="every", all seeds
        unique_algs = all_df["alg"].unique().to_list()
        bars = [(alg, None, "every", None) for alg in unique_algs]

    # Collect agg_data for each bar
    agg_data = []
    for bar_alg, bar_aperture, bar_sample_type, bar_seeds in bars:
        bar_agg_data = []
        # Filter data for this bar
        bar_df = all_df.filter(pl.col("sample_type") == bar_sample_type)
        if bar_seeds is not None:
            bar_df = bar_df.filter(pl.col("seed").is_in(bar_seeds))
        if bar_aperture is not None:
            bar_df = bar_df.filter(pl.col("aperture") == bar_aperture)

        # Filter to the alg
        bar_df = bar_df.filter(pl.col("alg") == bar_alg)

        # Filter data to frame == target_frame (or max frame if less)
        max_frame = bar_df.select(pl.col("frame").max()).item()
        filtered_df = bar_df.filter(pl.col("frame") == max_frame)

        print(
            f"For bar {bar_alg} aperture {bar_aperture} {bar_sample_type} seeds {bar_seeds}: filtered df shape {filtered_df.shape}"
        )

        # Aggregate mean_reward for this bar
        for group_key, df in filtered_df.group_by(
            ["alg_base", "aperture", "alg", "seed"]
        ):
            alg_base, aperture, alg, seed = group_key
            if metric in df.columns:
                mean_val = df[metric].item()
                if mean_val is not None and mean_val == mean_val:
                    bar_agg_data.append(
                        {
                            "bar_alg": bar_alg,
                            "bar_sample_type": bar_sample_type,
                            "bar_seeds": bar_seeds,
                            "bar_seeds_str": ",".join(map(str, bar_seeds))
                            if bar_seeds
                            else "",
                            "bar_aperture": bar_aperture,
                            "bar_aperture_str": str(bar_aperture) if bar_aperture else "",
                            "alg_base": alg_base,
                            "aperture": aperture,
                            "alg": alg,
                            "seed": seed,
                            metric: mean_val,
                        }
                    )

        if not bar_agg_data:
            raise ValueError(
                f"No data available for bar: {bar_alg}, aperture: {bar_aperture}, {bar_sample_type}, seeds: {bar_seeds}"
            )

        agg_data.extend(bar_agg_data)

    agg_df = pl.DataFrame(agg_data)

    # Prepare data for plotting with bootstrapped confidence intervals
    bar_labels = []
    mean_rewards = []
    errors_lower = []
    errors_upper = []

    colors = select_colors(len(bars), override="vibrant")

    for _i, (bar_alg, bar_aperture, bar_sample_type, bar_seeds) in enumerate(bars):
        # Generate label for this bar
        alg_label = str(LABEL_MAP.get(bar_alg, bar_alg))
        if bar_aperture is not None:
            alg_label += f"\n(aperture {bar_aperture})"
        if bar_seeds is not None and len(bar_seeds) == 1:
            alg_label += f"\n(seed {bar_seeds[0]})"
        elif bar_seeds is not None:
            alg_label += f"\n(seeds {','.join(map(str, bar_seeds))})"
        bar_labels.append(alg_label)

        # Get all seed values for this bar
        bar_seeds_str = ",".join(map(str, bar_seeds)) if bar_seeds else ""
        bar_aperture_str = str(bar_aperture) if bar_aperture else ""
        seed_values = agg_df.filter(
            (pl.col("bar_alg") == bar_alg)
            & (pl.col("bar_sample_type") == bar_sample_type)
            & (pl.col("bar_seeds_str") == bar_seeds_str)
            & (pl.col("bar_aperture_str") == bar_aperture_str)
        )[metric].to_numpy()

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

        mean_rewards.append(mean_val)
        errors_lower.append(lower_err)
        errors_upper.append(upper_err)

    # Plot bar chart
    fig, ax = plt.subplots(layout="constrained")

    x = np.arange(len(bar_labels))
    width = 0.8  # Width of each bar

    ax.bar(
        x,
        mean_rewards,
        width,
        color=colors,
        yerr=[errors_lower, errors_upper],
        capsize=3
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel(format_metric_name(metric))
    ax.set_title(f"{env} - {format_metric_name(metric)}")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=plot_name or f"{env}_{metric}_bar",
        save_type=save_type,
        f=fig,
        width=2,
        height_ratio=2/len(bar_labels),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mean reward bar plots from processed data"
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
        help="Bar specifications in format 'alg|aperture|sample_type|seeds' where aperture and seeds are comma-separated or empty (e.g., 'DQN|45|slice_1000000_1000_500|0,1')",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=None,
        help="Custom plot name (default: {env}_{metric}_bar)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_reward",
        help="Metric to plot (default: mean_reward)",
    )
    args = parser.parse_args()

    # Parse bars
    bars = None
    if args.bars:
        bars = []
        for bar_spec in args.bars:
            parts = bar_spec.split("|")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid bar spec: {bar_spec}. Expected format 'alg|aperture|sample_type|seeds'"
                )
            alg, aperture_str, sample_type, seeds_str = parts
            if aperture_str:
                aperture = int(aperture_str.strip())
            else:
                aperture = None
            if seeds_str:
                seeds = [int(s.strip()) for s in seeds_str.split(",")]
            else:
                seeds = None
            bars.append((alg, aperture, sample_type, seeds))

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.save_type,
        bars,
        args.plot_name,
        args.metric,
    )

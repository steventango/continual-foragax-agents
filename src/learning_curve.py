import argparse
import json
import os
import sys
from itertools import chain
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition
from scipy.stats import bootstrap

from utils.constants import LABEL_MAP
from utils.plotting import label_lines, select_colors

setDefaultConference("jmlr")
setFonts(20)


def main(
    experiment_path: Path,
    normalize: str | None,
    save_type: str = "pdf",
    filter_alg_apertures: list[str] | None = None,
    plot_name: str | None = None,
    ylim: tuple[float, float] | None = None,
    auto_label: bool = False,
    sample_type: str = "every",
    metric: str = "ewm_reward",
):
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
        / "data.parquet"
    )

    # Load processed data
    all_df = pl.read_parquet(data_path)
    print(all_df)

    # Filter by sample_type
    print(list(all_df["sample_type"].unique()))
    all_df = all_df.filter(pl.col("sample_type") == sample_type)

    # Derive additional columns
    all_df = all_df.with_columns(
        pl.col("alg").str.replace(r"_frozen_.*", "").alias("alg_base"),
        pl.when(pl.col("alg").str.contains("_frozen"))
        .then(pl.col("alg").str.extract(r"_frozen_(.*)", 1))
        .otherwise(None)
        .alias("freeze_steps_str"),
    )

    algs = all_df["alg"].unique()
    print("Algs:", list(algs))

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

    # Compute metadata from df
    main_algs = sorted(
        all_df.filter(pl.col("aperture").is_not_null())["alg_base", "aperture"]
        .unique()
        .iter_rows(named=False)
    )
    print("Main algs:", main_algs)
    env = all_df["env"][0]

    # Assign unique colors to each (alg, aperture) combination
    alg_aperture_df = (
        all_df.filter(pl.col("aperture").is_not_null())
        .select("alg", "aperture")
        .unique()
        .sort(["alg", "aperture"])
    )
    pairs_list = list(alg_aperture_df.iter_rows(named=False))
    
    # For baselines without aperture
    baseline_algs = all_df.filter(pl.col("aperture").is_null())["alg"].unique()
    baseline_pairs = [(alg, None) for alg in baseline_algs]
    
    # All unique combinations
    all_pairs = pairs_list + baseline_pairs
    n_colors = len(all_pairs)
    if n_colors > 0:
        color_list = select_colors(n_colors, override="vibrant")
        pair_to_color = dict(zip(all_pairs, color_list, strict=True))
    else:
        pair_to_color = {}
    
    # Assign COLORS
    COLORS = pair_to_color
    baseline_ys_dict = {}
    normalized = False
    if normalize is not None:
        baseline_df = all_df.filter(pl.col("alg_base") == normalize)
        if baseline_df.height > 0:
            baseline_df = (
                baseline_df.sort("seed")
                .group_by("seed")
                .agg(pl.col("frame"), pl.col(metric))
            )
            baseline_ys_dict = {
                row["seed"]: np.array(row[metric])
                for row in baseline_df.iter_rows(named=True)
            }
            normalized = True
        else:
            print(
                f"Warning: No data found for baseline algorithm '{normalize}'. Normalization skipped."
            )

    dd = data_definition(
        hyper_cols=[],  # will be set from df
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    num_seeds = all_df.select(pl.col(dd.seed_col).max()).item() + 1

    ncols = 1
    # Single subplot for mean plots
    nrows_mean = 1
    ncols_mean = 1
    # Create separate figures for mean and seeds
    fig_mean, axs_mean = plt.subplots(
        nrows_mean,
        ncols_mean,
        sharex=True,
        sharey="all",
        layout="constrained",
        squeeze=False,
    )
    fig_seeds, axs_seeds = plt.subplots(
        num_seeds, ncols, sharex=True, sharey="all", layout="constrained", squeeze=False
    )

    for group_key, df in all_df.group_by(["alg_base", "aperture", "alg"]):
        alg_base = group_key[0]
        aperture = group_key[1]
        alg = group_key[2]
        print(alg_base, aperture, alg)

        if "sweep" in str(experiment_path):
            # Check if best configuration exists
            if aperture is not None:
                best_configuration_path = (
                    experiment_path / "hypers" / str(aperture) / f"{alg}.json"
                )
            else:
                continue
            if best_configuration_path.exists():
                with open(best_configuration_path) as f:
                    best_configuration = json.load(f)

                # Filter df to only include rows matching best configuration
                for param, value in best_configuration.items():
                    if param in df.columns:
                        df = df.filter(pl.col(param) == value)

        df = df.sort(dd.seed_col).group_by(dd.seed_col).agg(dd.time_col, metric)

        try:
            xs = np.stack(df["frame"].to_numpy())  # type: ignore
        except Exception as e:
            print(f"Skipping {alg} due to error in stacking frames: {e}")
            continue
        ys = np.stack(df[metric].to_numpy())  # type: ignore
        mask = xs[0] > 1000
        xs = xs[:, mask]
        ys = ys[:, mask]
        print(ys.shape)
        assert np.all(np.isclose(xs[0], xs))

        # Normalize to baseline
        if baseline_ys_dict:
            for i in range(len(ys)):
                seed = df["seed"][i]
                if seed in baseline_ys_dict:
                    baseline = baseline_ys_dict[seed][mask]
                    ys[i] /= baseline

        # Compute bootstrap statistics using scipy on the entire array
        sample_stat = np.mean(ys, axis=0)
        bs_result = bootstrap(
            (ys,),
            lambda x, axis=0: np.mean(x, axis=axis),
            axis=0,
            confidence_level=0.95,
            n_resamples=10000,
        )
        ci_low = bs_result.confidence_interval.low
        ci_high = bs_result.confidence_interval.high

        # Clip res to [-2, 2] for normalized plots
        if normalized:
            sample_stat = np.where(
                (sample_stat >= -2) & (sample_stat <= 2),
                sample_stat,
                np.nan,
            )
            ci_low = np.where((ci_low >= -2) & (ci_low <= 2), ci_low, np.nan)
            ci_high = np.where((ci_high >= -2) & (ci_high <= 2), ci_high, np.nan)
        else:
            pass  # no clipping needed

        freeze_steps_str = alg.split("_frozen")[1] if "_frozen" in alg else None

        linestyle = "-"

        # Create label for auto-labeling
        alg_label = str(LABEL_MAP.get(alg_base, alg_base))
        plot_label_parts = [alg_label]
        if freeze_steps_str:
            plot_label_parts.append(f"(Frozen @ {freeze_steps_str.lstrip('_')})")
        if aperture is not None:
            plot_label_parts.append(f"(FOV {aperture})")
        plot_label = " ".join(plot_label_parts)

        # Plot
        if aperture is not None:
            color = COLORS[(alg, aperture)]
            # Plot mean on mean figure
            ax = axs_mean[0, 0]
            ax.plot(
                xs[0],
                sample_stat,
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=plot_label,
            )
            if len(ys) >= 5:
                ax.fill_between(xs[0], ci_low, ci_high, color=color, alpha=0.2)
            # Plot each seed on seeds figure
            for i in range(len(ys)):
                ax = axs_seeds[i, 0]
                ax.plot(
                    xs[0],
                    ys[i],
                    color=color,
                    linewidth=0.5,
                    linestyle=linestyle,
                    label=plot_label,
                )
        else:
            color = COLORS[(alg, None)]
            # Plot mean on mean figure
            ax = axs_mean[0, 0]
            if alg_base == "Search-Oracle" and normalized:
                ax.axhline(
                    1,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                    label=plot_label,
                )
            else:
                ax.plot(
                    xs[0],
                    sample_stat,
                    color=color,
                    linewidth=1.0,
                    linestyle=linestyle,
                    label=plot_label,
                )
                if len(ys) >= 5:
                    ax.fill_between(xs[0], ci_low, ci_high, color=color, alpha=0.2)
            # Plot each seed on seeds figure
            color = COLORS[(alg, None)]
            for i in range(len(ys)):
                ax = axs_seeds[i, 0]
                if alg_base == "Search-Oracle" and normalized:
                    ax.axhline(
                        1,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.5,
                        label=plot_label,
                    )
                else:
                    ax.plot(
                        xs[0],
                        ys[i],
                        color=color,
                        linewidth=0.5,
                        linestyle=linestyle,
                        label=plot_label,
                    )

    # Format mean axes
    ylabel = "Normalized Average Reward" if normalized else "Average Reward"
    ax = axs_mean[0, 0]
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time steps")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format seeds axes
    ylabel = "Normalized Average Reward" if normalized else "Average Reward"
    for i in range(num_seeds):
        ax = axs_seeds[i, 0]
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_ylabel(ylabel)
        if i == num_seeds - 1:
            ax.set_xlabel("Time steps")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in chain(axs_mean.flatten(), axs_seeds.flatten()):
        if not ax.get_lines():
            ax.set_visible(False)
            continue

    # Collect unique algorithms for legend
    label_to_color = {}
    # Collect from groups
    for group_key, _ in all_df.group_by(["alg_base", "aperture", "alg"]):
        alg_base = group_key[0]
        aperture = group_key[1]
        alg = group_key[2]
        freeze_steps_str = alg.split("_frozen")[1] if "_frozen" in alg else None
        alg_label = str(LABEL_MAP.get(alg_base, alg_base))
        plot_label_parts = [alg_label]
        if freeze_steps_str:
            plot_label_parts.append(f"(Frozen @ {freeze_steps_str.lstrip('_')})")
        if aperture is not None:
            plot_label_parts.append(f"(FOV {aperture})")
        plot_label = " ".join(plot_label_parts)
        if aperture is not None:
            color = COLORS[(alg, aperture)]
        else:
            color = COLORS[(alg, None)]
        label_to_color[plot_label] = color

    # Order legend elements according to filter_alg_apertures if provided
    if filter_alg_apertures:
        ordered_labels = []
        for pair in filter_alg_apertures:
            if ":" in pair:
                alg, aperture_str = pair.split(":", 1)
                try:
                    aperture = int(aperture_str)
                except ValueError:
                    aperture = aperture_str
                label = f"{LABEL_MAP.get(alg, alg)} (FOV {aperture})"
            else:
                label = LABEL_MAP.get(pair, pair)
            if label in label_to_color:
                ordered_labels.append(label)
        # Add any remaining labels not in filter_alg_apertures
        for label in label_to_color:
            if label not in ordered_labels:
                ordered_labels.append(label)
    else:
        ordered_labels = sorted(label_to_color.keys())

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=label_to_color[label],
            lw=2,
            label=label,
            linestyle="-",
        )
        for label in ordered_labels
    ]

    # Set ylim if provided
    if ylim is not None:
        for ax in chain(axs_mean.flatten(), axs_seeds.flatten()):
            if ax.get_lines():  # Only set if there are lines
                ax.set_ylim(ylim)

    # Handle legend or auto labeling
    if auto_label:
        # Use automatic label annotation instead of legend for mean plot
        for ax in axs_mean.flatten():
            if ax.get_lines():
                label_lines(ax, offset_range=(12, 18))
        fig_mean.suptitle(env)
        # Use legend for seeds plot
        fig_seeds.suptitle(f"{env} - Individual Seeds")
        fig_seeds.legend(
            handles=legend_elements, loc="outside center right", frameon=False
        )
    else:
        fig_mean.suptitle(env)
        fig_mean.legend(
            handles=legend_elements, loc="outside center right", frameon=False
        )

        fig_seeds.suptitle(f"{env} - Individual Seeds")
        fig_seeds.legend(
            handles=legend_elements, loc="outside center right", frameon=False
        )

    base_name = plot_name if plot_name else env

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=f"{base_name}{'_normalized' if normalized else ''}",
        save_type=save_type,
        f=fig_mean,
        width=ncols_mean if ncols_mean > 1 else (1 if auto_label else 2),
        height_ratio=2 / 3 * nrows_mean / ncols_mean
        if ncols_mean > 1
        else (2 / 3 * nrows_mean if auto_label else nrows_mean / 3),
    )
    save(
        save_path=f"{experiment_path}/plots",
        plot_name=f"{base_name}_seeds{('_normalized' if normalized else '')}",
        save_type=save_type,
        f=fig_seeds,
        width=ncols if ncols > 1 else (1 if auto_label else 2),
        height_ratio=(num_seeds / ncols) * (2 / 3)
        if ncols > 1
        else (2 / 3 * num_seeds if auto_label else num_seeds / 3),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot learning curves from processed data"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the experiment directory",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Algorithm to normalize results to before bootstrapping (default: no normalization)",
    )
    parser.add_argument(
        "--save-type",
        type=str,
        default="pdf",
        help="File format to save the plots (default: pdf)",
    )
    parser.add_argument(
        "--filter-alg-apertures",
        nargs="*",
        default=None,
        help="List of algorithm:aperture pairs to show (default: all). Use 'alg' for alg-only filtering.",
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
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to plot (default: every)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ewm_reward",
        help="Metric to plot (default: ewm_reward)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.path).resolve()
    main(
        experiment_path,
        args.normalize,
        args.save_type,
        args.filter_alg_apertures,
        args.plot_name,
        args.ylim,
        args.auto_label,
        args.sample_type,
        args.metric,
    )

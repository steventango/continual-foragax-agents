import argparse
import json
import math
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
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
)

from utils.constants import LABEL_MAP
from utils.plotting import label_lines, select_colors

setDefaultConference("jmlr")
setFonts(20)


def main(
    experiment_path: Path,
    normalize: str | None,
    save_type: str = "pdf",
    filter_algs: list[str] | None = None,
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

    # Filter by sample_type
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
    print("Algs:", algs)

    # Filter algorithms if specified
    if filter_algs:
        all_df = all_df.filter(pl.col("alg").is_in(filter_algs))

    # Compute metadata from df
    main_algs = sorted(
        all_df.filter(pl.col("aperture").is_not_null())["alg_base", "aperture"]
        .unique()
        .iter_rows(named=False)
    )
    print("Main algs:", main_algs)
    env = all_df["env"][0]

    # Collect all color keys and assign colors based on algorithm type
    all_color_keys = set(all_df["alg"].unique())

    def get_type_key(alg):
        alg_base = alg.split("_frozen")[0] if "_frozen" in alg else alg
        if alg_base.startswith("Search"):
            return alg
        elif "_frozen" in alg:
            freeze_steps_str = alg.split("_frozen_")[1]
            if freeze_steps_str == "1M":
                return "frozen_1M"
            elif freeze_steps_str == "5M":
                return "frozen_5M"
            else:
                return "frozen_other"
        else:
            return "base"

    type_keys = set(get_type_key(alg) for alg in all_color_keys)
    n_colors = len(type_keys)
    color_list = select_colors(n_colors)
    type_to_color = dict(zip(sorted(type_keys), color_list, strict=True))
    COLORS = {alg: type_to_color[get_type_key(alg)] for alg in all_color_keys}

    # Compute baseline_ys_dict for normalization
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

    ncols = max(len(main_algs), 1)
    # Calculate nrows and ncols for mean plots to be close to square
    n_plots = len(main_algs)
    if n_plots == 0:
        nrows_mean = 1
        ncols_mean = 1
    else:
        nrows_mean = int(math.sqrt(n_plots))
        ncols_mean = math.ceil(n_plots / nrows_mean)
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

        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=ys,
            statistic=Statistic.mean,
            iterations=10000,
        )

        # Clip res to [-2, 2] for normalized plots
        if normalized:
            sample_stat = np.where(
                (res.sample_stat >= -2) & (res.sample_stat <= 2),
                res.sample_stat,
                np.nan,
            )
            ci_low = np.where((res.ci[0] >= -2) & (res.ci[0] <= 2), res.ci[0], np.nan)
            ci_high = np.where((res.ci[1] >= -2) & (res.ci[1] <= 2), res.ci[1], np.nan)
        else:
            sample_stat = res.sample_stat
            ci_low = res.ci[0]
            ci_high = res.ci[1]

        freeze_steps_str = alg.split("_frozen")[1] if "_frozen" in alg else None

        linestyle = "-"

        # Create label for auto-labeling
        alg_label = str(LABEL_MAP.get(alg_base, alg_base))
        if freeze_steps_str:
            plot_label = f"{alg_label} (Frozen @ {freeze_steps_str.lstrip('_')})"
        else:
            plot_label = alg_label

        # Plot
        if aperture is not None:
            col_linear = main_algs.index((alg_base, aperture))
            row = col_linear // ncols_mean
            col = col_linear % ncols_mean
            color = COLORS[alg]
            # Plot mean on mean figure
            ax = axs_mean[row, col]
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
                ax = axs_seeds[i, col_linear]
                ax.plot(
                    xs[0],
                    ys[i],
                    color=color,
                    linewidth=0.5,
                    linestyle=linestyle,
                    label=plot_label,
                )
        else:
            # Plot mean on mean figure, all subplots
            color = COLORS[alg]
            for row in range(nrows_mean):
                for col in range(ncols_mean):
                    ax = axs_mean[row, col]
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
                            ax.fill_between(
                                xs[0], ci_low, ci_high, color=color, alpha=0.2
                            )
            # Plot each seed on seeds figure, all columns
            color = COLORS[alg]
            for i in range(len(ys)):
                for col in range(ncols):
                    ax = axs_seeds[i, col]
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

    # Set titles and formatting
    for i, (alg_base, aperture_val) in enumerate(main_algs):
        alg_label = LABEL_MAP.get(alg_base, alg_base)
        if alg_label is None:
            alg_label = alg_base
        row = i // ncols_mean
        col = i % ncols_mean
        # Title for mean figure
        title = alg_label
        if aperture_val is not None:
            title += f" ({aperture_val})"
        axs_mean[row, col].set_title(title)
        # Title for seeds figure (first row)
        axs_seeds[0, i].set_title(title)

    # Format mean axes
    ylabel = "Normalized Average Reward" if normalized else "Average Reward"
    for row in range(nrows_mean):
        for col in range(ncols_mean):
            ax = axs_mean[row, col]
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            if col == 0:  # Only leftmost columns get ylabel
                ax.set_ylabel(ylabel)
            if row == nrows_mean - 1:  # Only bottom row gets xlabel
                ax.set_xlabel("Time steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Format seeds axes
    ylabel = "Normalized Average Reward" if normalized else "Average Reward"
    for i in range(num_seeds):
        for j in range(ncols):
            ax = axs_seeds[i, j]
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            if j == 0:
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
    unique_algs = sorted(all_df["alg"].unique())
    label_to_color = {}
    for alg in unique_algs:
        type_key = get_type_key(alg)
        if type_key in ["base", "frozen_1M", "frozen_5M"]:
            label = {
                "base": "Base Algorithm",
                "frozen_1M": "Frozen @ 1M",
                "frozen_5M": "Frozen @ 5M",
            }[type_key]
            label_to_color[label] = COLORS[alg]
        elif type_key == "baseline":
            label = LABEL_MAP.get(alg, alg)
            label_to_color[label] = COLORS[alg]
        else:
            label = LABEL_MAP.get(alg, alg)
            label_to_color[label] = COLORS[alg]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            label=label,
            linestyle="-",
        )
        for label, color in sorted(label_to_color.items())
    ]

    # Set ylim if provided
    if ylim is not None:
        for ax in chain(axs_mean.flatten(), axs_seeds.flatten()):
            if ax.get_lines():  # Only set if there are lines
                ax.set_ylim(ylim)

    # Handle legend or auto labeling
    if auto_label:
        # Use automatic label annotation instead of legend
        for ax in chain(axs_mean.flatten(), axs_seeds.flatten()):
            if ax.get_lines():
                label_lines(ax, offset_range=(12, 18))
        fig_mean.suptitle(env)
        fig_seeds.suptitle(f"{env} - Individual Seeds")
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
        args.filter_algs,
        args.plot_name,
        args.ylim,
        args.auto_label,
        args.sample_type,
        args.metric,
    )

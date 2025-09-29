import argparse
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
from utils.plotting import select_colors

setDefaultConference("jmlr")
setFonts(20)


def main(experiment_path: Path, normalize: str | None):
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments"))
        / "data.parquet"
    )

    # Load processed data
    all_df = pl.read_parquet(data_path)

    # Derive additional columns
    all_df = all_df.with_columns(
        pl.col("alg").str.split("_frozen").list.get(0).alias("alg_base"),
        pl.col("alg").str.contains("frozen").alias("frozen"),
    )

    # Compute metadata from df
    unique_alg_bases = sorted(all_df["alg_base"].unique())
    main_algs = sorted(
        all_df.filter(pl.col("aperture").is_not_null())["alg_base", "aperture"]
        .unique()
        .iter_rows(named=False)
    )
    print("Main algs:", main_algs)
    env = all_df["env"][0]

    # Collect all color keys
    all_color_keys = set(unique_alg_bases)
    n_colors = len(all_color_keys)
    color_list = select_colors(n_colors)
    sorted_keys = sorted(all_color_keys)
    COLORS = dict(zip(sorted_keys, color_list, strict=True))
    SINGLE = unique_alg_bases
    metric = "ewm_reward"

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

        color = "grey" if "frozen" in alg else COLORS[alg_base]
        linestyle = "--" if "frozen" in alg else "-"

        # Plot
        if aperture is not None:
            col_linear = main_algs.index((alg_base, aperture))
            row = col_linear // ncols_mean
            col = col_linear % ncols_mean
            # Plot mean on mean figure
            ax = axs_mean[row, col]
            ax.plot(
                xs[0],
                sample_stat,
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
            )
            if len(ys) >= 5:
                ax.fill_between(xs[0], ci_low, ci_high, color=color, alpha=0.2)
            # Plot each seed on seeds figure
            for i in range(len(ys)):
                ax = axs_seeds[i, col_linear]
                ax.plot(xs[0], ys[i], color=color, linewidth=0.5, linestyle=linestyle)
        else:
            # Plot mean on mean figure, all subplots
            for row in range(nrows_mean):
                for col in range(ncols_mean):
                    ax = axs_mean[row, col]
                    if alg_base == "Search-Oracle" and normalized:
                        ax.axhline(1, color=color, linestyle=linestyle, linewidth=1.0)
                    else:
                        ax.plot(
                            xs[0],
                            sample_stat,
                            color=color,
                            linewidth=1.0,
                            linestyle=linestyle,
                        )
                        if len(ys) >= 5:
                            ax.fill_between(
                                xs[0], ci_low, ci_high, color=color, alpha=0.2
                            )
            # Plot each seed on seeds figure, all columns
            for i in range(len(ys)):
                for col in range(ncols):
                    ax = axs_seeds[i, col]
                    if alg_base == "Search-Oracle" and normalized:
                        ax.axhline(1, color=color, linestyle=linestyle, linewidth=0.5)
                    else:
                        ax.plot(
                            xs[0],
                            ys[i],
                            color=color,
                            linewidth=0.5,
                            linestyle=linestyle,
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

    legend_elements = []
    for color_key in SINGLE:
        alg_label = str(LABEL_MAP.get(color_key, color_key))
        # Normal version
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=COLORS[color_key],
                lw=2,
                label=alg_label,
                linestyle="-",
            )
        )
        # Frozen version if exists
        if (
            all_df.filter(pl.col("alg_base") == color_key)
            .filter(pl.col("frozen"))
            .height
            > 0
        ):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="grey",
                    lw=2,
                    label=alg_label + " (Frozen)",
                    linestyle="--",
                )
            )

    # Sort legend elements by label
    legend_elements.sort(key=lambda x: x.get_label())

    fig_mean.suptitle(env)
    fig_mean.legend(handles=legend_elements, loc="outside center right", frameon=False)

    fig_seeds.suptitle(f"{env} - Individual Seeds")
    fig_seeds.legend(handles=legend_elements, loc="outside center right", frameon=False)

    path_plots = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path_plots}/plots",
        plot_name=f"{env}{'_normalized' if normalized else ''}",
        save_type="pdf",
        f=fig_mean,
        width=ncols_mean if ncols_mean > 1 else 2,
        height_ratio=2 / 3 * nrows_mean / ncols_mean if ncols_mean > 1 else nrows_mean / 3
    )
    save(
        save_path=f"{path_plots}/plots",
        plot_name=f"{env}_seeds{('_normalized' if normalized else '')}",
        save_type="pdf",
        f=fig_seeds,
        width=ncols if ncols > 1 else 2,
        height_ratio=(num_seeds / ncols) * (2 / 3) if ncols > 1 else num_seeds / 3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot learning curves from processed data"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the experiment directory",
        default="experiments/E63-baselines/foragax/ForagaxTwoBiome-v8",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Algorithm to normalize results to before bootstrapping (default: no normalization)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.path)
    main(experiment_path, args.normalize)

import argparse
import math
import os
import sys
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


def main(experiment_path: Path, trace_exponent: int, save_type: str = "pdf"):
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
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
    main_algs = sorted(all_df["alg_base", "aperture"].unique().iter_rows(named=False))
    print("Main algs:", main_algs)
    env = all_df["env"][0]

    trace_metrics = [
        f"morel_trace_{trace_exponent}",
        f"oyster_trace_{trace_exponent}",
        f"deathcap_trace_{trace_exponent}",
    ]
    mushroom_names = ["Morel", "Oyster", "Deathcap"]
    metric_colors = dict(
        zip(trace_metrics, select_colors(len(trace_metrics)), strict=True)
    )

    dd = data_definition(
        hyper_cols=[],
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
        alg_base, aperture, alg = group_key
        print(f"Plotting: {alg}")

        is_frozen = df["frozen"][0]
        linestyle = "--" if is_frozen else "-"

        for metric in trace_metrics:
            metric_df = (
                df.sort(dd.seed_col).group_by(dd.seed_col).agg(dd.time_col, metric)
            )
            if metric_df.is_empty() or metric_df[metric].is_null().all():
                continue

            try:
                xs = np.stack(metric_df["frame"].to_list())
            except Exception as e:
                print(f"Skipping {alg} for metric {metric} due to error: {e}")
                continue

            ys = np.stack(metric_df[metric].to_list())
            mask = xs[0] > 1000
            xs, ys = xs[:, mask], ys[:, mask]

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=10000,
            )

            color = metric_colors[metric]

            col_linear = main_algs.index((alg_base, aperture))
            row = col_linear // ncols_mean
            col = col_linear % ncols_mean

            # Plot mean on mean figure
            ax = axs_mean[row, col]
            ax.plot(
                xs[0],
                res.sample_stat,
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
            )
            if len(ys) >= 5:
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)

            # Plot each seed on seeds figure
            for i in range(len(ys)):
                ax = axs_seeds[i, col_linear]
                ax.plot(
                    xs[0], ys[i], color=color, linewidth=0.5, linestyle=linestyle
                )

    # Set titles and formatting for mean plot
    for i, (alg_base, aperture_val) in enumerate(main_algs):
        alg_label = LABEL_MAP.get(alg_base, alg_base)
        row = i // ncols_mean
        col = i % ncols_mean
        title = (
            f"{alg_label} ({aperture_val})" if aperture_val is not None else alg_label
        )
        axs_mean[row, col].set_title(title)

    # Set titles and formatting for seeds plot
    for i, (alg_base, aperture_val) in enumerate(main_algs):
        alg_label = LABEL_MAP.get(alg_base, alg_base)
        title = (
            f"{alg_label} ({aperture_val})" if aperture_val is not None else alg_label
        )
        axs_seeds[0, i].set_title(title)

    # Format mean axes
    for row in range(nrows_mean):
        for col in range(ncols_mean):
            ax = axs_mean[row, col]
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            if col == 0:
                ax.set_ylabel(f"Mushroom Trace (1e-{trace_exponent})")
            if row == nrows_mean - 1:
                ax.set_xlabel("Time steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Format seeds axes
    for i in range(num_seeds):
        for j in range(ncols):
            ax = axs_seeds[i, j]
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            if j == 0:
                ax.set_ylabel(f"Mushroom Trace (1e-{trace_exponent})")
            if i == num_seeds - 1:
                ax.set_xlabel("Time steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axs_mean.flatten():
        if not ax.get_lines():
            ax.set_visible(False)

    for ax in axs_seeds.flatten():
        if not ax.get_lines():
            ax.set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=metric_colors[m], lw=2, label=n)
        for m, n in zip(trace_metrics, mushroom_names, strict=True)
    ]
    fig_mean.suptitle(f"{env}")
    fig_mean.legend(handles=legend_elements, loc="outside center right", frameon=False)

    fig_seeds.suptitle(f"{env} - Individual Seeds")
    fig_seeds.legend(handles=legend_elements, loc="outside center right", frameon=False)

    save(
        save_path=f"{experiment_path}/plots",
        plot_name=f"{env}_mushroom_trace_e{trace_exponent}",
        save_type=save_type,
        f=fig_mean,
        width=ncols_mean if ncols_mean > 1 else 2,
        height_ratio=2 / 3 * nrows_mean / ncols_mean
        if ncols_mean > 1
        else nrows_mean / 3,
    )
    save(
        save_path=f"{experiment_path}/plots",
        plot_name=f"{env}_mushroom_trace_seeds_e{trace_exponent}",
        save_type=save_type,
        f=fig_seeds,
        width=ncols if ncols > 1 else 2,
        height_ratio=(num_seeds / ncols) * (2 / 3) if ncols > 1 else num_seeds / 3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mushroom trace metrics from processed data"
    )
    parser.add_argument("path", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--trace-exponent",
        type=int,
        default=3,
        help="The exponent for the trace metric (e.g., 3 for 1e-3)",
    )
    parser.add_argument(
        "--save-type",
        type=str,
        default="pdf",
        help="File format to save the plots (default: pdf)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.path).resolve()
    main(experiment_path, args.trace_exponent, args.save_type)

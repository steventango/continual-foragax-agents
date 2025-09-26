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


def main(experiment_path: Path):
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
        all_df.filter(pl.col("aperture").is_not_null())["alg_base"].unique()
    )
    env = all_df["env"][0]

    # Collect all color keys
    all_color_keys = set(unique_alg_bases)
    n_colors = len(all_color_keys)
    color_list = select_colors(n_colors)
    sorted_keys = sorted(all_color_keys)
    COLORS = dict(zip(sorted_keys, color_list, strict=True))
    SINGLE = unique_alg_bases
    metric = "ewm_reward"

    dd = data_definition(
        hyper_cols=[],  # will be set from df
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    num_seeds = all_df.select(pl.col(dd.seed_col).max()).item() + 1

    ncols = len(main_algs)
    nrows = 1 + num_seeds
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey="all", layout="constrained", squeeze=False
    )

    for group_key, df in all_df.group_by(["alg_base", "aperture", "alg"]):
        alg_base = group_key[0]
        aperture = group_key[1]
        alg = group_key[2]
        print(alg)

        df = df.sort(dd.seed_col).group_by(dd.seed_col).agg(dd.time_col, metric)

        xs = np.stack(df["frame"].to_numpy())  # type: ignore
        ys = np.stack(df[metric].to_numpy())  # type: ignore
        mask = xs[0] > 1000
        xs = xs[:, mask]
        ys = ys[:, mask]
        print(ys.shape)
        assert np.all(np.isclose(xs[0], xs))

        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=ys,
            statistic=Statistic.mean,
            iterations=10000,
        )

        color = "grey" if "frozen" in alg else COLORS[alg_base]
        linestyle = "--" if "frozen" in alg else "-"

        # Plot
        if aperture is not None:
            col = main_algs.index(alg_base)
            # Plot mean on row 0
            ax = axs[0, col]
            ax.plot(
                xs[0],
                res.sample_stat,
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
            )
            if len(ys) >= 5:
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)
            # Plot each seed on subsequent rows
            for i in range(len(ys)):
                ax = axs[1 + i, col]
                ax.plot(xs[0], ys[i], color=color, linewidth=0.5, linestyle=linestyle)
        else:
            # Plot mean on row 0, all columns
            for col in range(ncols):
                ax = axs[0, col]
                ax.plot(
                    xs[0],
                    res.sample_stat,
                    color=color,
                    linewidth=1.0,
                    linestyle=linestyle,
                )
                if len(ys) >= 5:
                    ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)
            # Plot each seed on subsequent rows, all columns
            for i in range(len(ys)):
                for col in range(ncols):
                    ax = axs[1 + i, col]
                    ax.plot(
                        xs[0], ys[i], color=color, linewidth=0.5, linestyle=linestyle
                    )

    # Set titles and formatting
    for col in range(ncols):
        ax = axs[0, col]
        alg_base = main_algs[col]
        alg_label = LABEL_MAP.get(alg_base, alg_base)
        ax.set_title(f"{alg_label}")

    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i, j]
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            if j == 0:
                ax.set_ylabel("Average Reward")
            if i == nrows - 1:
                ax.set_xlabel("Time steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axs.flatten():
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

    fig.suptitle(env)
    fig.legend(handles=legend_elements, loc="outside center right", frameon=False)

    path_plots = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path_plots}/plots",
        plot_name=env,
        save_type="pdf",
        f=fig,
        width=2 * ncols,
        height_ratio=(nrows / ncols) * (1 / 3),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot learning curves from processed data"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the experiment directory",
        default="experiments/E46-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v7",
    )
    args = parser.parse_args()

    experiment_path = Path(args.path)
    main(experiment_path)

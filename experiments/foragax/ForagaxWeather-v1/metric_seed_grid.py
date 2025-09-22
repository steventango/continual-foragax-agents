import os
import re
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
import polars as pl
import tol_colors as tc
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.constants import BIOME_COLORS
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

colorset = tc.colorsets["high_contrast"]
alg_colors = tc.colorsets["bright"]


METRICS_TO_PLOT = [
    "ewm_reward",
    "loss",
    "squared_td_error",
    "abs_td_error",
    "weight_change",
    "occupancy_combined",
]

ENV_MAP = {"ForagaxTwoBiomeSmall": "ForagaxWeather-v1"}


def get_base_alg(alg):
    # Remove the freeze part: _Freeze_{number}
    base = re.sub(r"_Freeze_[^_]*", "", alg)
    return base


def get_freeze_num(alg):
    if "_Freeze_" not in alg:
        return 0
    match = re.search(r"_Freeze_([^_]*)", alg)
    if match:
        num_str = match.group(1)
        if num_str.endswith("k"):
            return int(num_str[:-1]) * 1000
        elif num_str.endswith("M"):
            return int(num_str[:-1]) * 1000000
        else:
            try:
                return int(num_str)
            except ValueError:
                return 0
    return 0


def plot_metric_seed_grid(ax, dfs, metric, seed_val, env, aperture):
    """Plot a single metric for a single seed on the given axis for multiple algorithms."""
    if not dfs:
        ax.set_visible(False)
        return

    if metric == "occupancy_combined":
        # For occupancy, plot only for the first algorithm to avoid overcrowding
        alg, df = next(iter(dfs.items()))
        if df is None or df.height == 0:
            ax.set_visible(False)
            return

        seed_df = df.filter(pl.col("seed") == seed_val)
        if seed_df.height == 0:
            ax.set_visible(False)
            return

        seed_df = seed_df.sort("frame")

        for occ_metric, color_key in zip(
            ["Morel_occupancy", "Oyster_occupancy", "Neither_occupancy"],
            ["Morel", "Oyster", "Neither"],
            strict=True,
        ):
            if occ_metric in seed_df.columns:
                frames = seed_df["frame"]
                values = seed_df[occ_metric]
                ax.plot(
                    frames,
                    values,
                    linewidth=1.0,
                    color=BIOME_COLORS[color_key],
                    label=color_key,
                )
            else:
                print(f"Column {occ_metric} not found in DataFrame.")  # Debugging
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.legend(fontsize=10, loc="upper right")
    else:
        plotted = False
        for idx, (alg, df) in enumerate(dfs.items()):
            if df is None or df.height == 0:
                continue

            seed_df = df.filter(pl.col("seed") == seed_val)
            if seed_df.height == 0:
                continue

            seed_df = seed_df.sort("frame")

            if metric not in seed_df.columns:
                continue

            frames = seed_df["frame"]
            values = seed_df[metric]
            if idx == 0:
                color = "black"
            else:
                color = alg_colors[(idx - 1) % len(alg_colors)]
            ax.plot(frames, values, linewidth=1.0, color=color, label=alg)
            plotted = True

        if not plotted:
            ax.set_visible(False)
            return

        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


if __name__ == "__main__":
    results = ResultCollection(
        Model=ExperimentModel,
        metrics=METRICS_TO_PLOT
        + ["Morel_occupancy", "Oyster_occupancy", "Neither_occupancy"],
    )
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    env_groups = {}
    for env_aperture, sub_results in results.groupby_directory(level=2):
        env, aperture = env_aperture.rsplit("-", 1)
        env = ENV_MAP.get(env, env)
        aperture = int(aperture)

        if env not in env_groups:
            env_groups[env] = {}
        env_groups[env][aperture] = sub_results

    for env, aperture_results in env_groups.items():
        apertures = sorted(aperture_results.keys())

        for aperture in apertures:
            sub_results = aperture_results.get(aperture, [])

            alg_groups = {}
            for alg_result in sub_results:
                alg = alg_result.filename
                if "Freeze" not in alg and alg not in {
                    "DQN",
                    "DQN_small_buffer",
                    "DQN_L2_Init",
                    "DQN_L2_Init_small_buffer",
                }:
                    continue
                base = get_base_alg(alg)
                if base not in alg_groups:
                    alg_groups[base] = []
                alg_groups[base].append(alg_result)

            print(f"Algorithm groups for {env}-{aperture}:")
            for base, results in alg_groups.items():
                algs = [r.filename for r in results]
                print(f"  {base}: {algs}")

            for base_alg, alg_results in alg_groups.items():
                print(f"Processing {env}-{aperture} {base_alg}")

                dfs = {}
                all_seeds = set()
                for alg_result in alg_results:
                    df = alg_result.load(sample=500)
                    if df is not None and df.height > 0:
                        alg = alg_result.filename
                        if "_Freeze_" in alg:
                            freeze_steps = get_freeze_num(alg)
                            for metric in METRICS_TO_PLOT:
                                if (
                                    metric not in ["ewm_reward", "occupancy_combined"]
                                    and metric in df.columns
                                ):
                                    df = df.with_columns(
                                        pl.when(pl.col("frame") > freeze_steps)
                                        .then(0)
                                        .otherwise(pl.col(metric))
                                        .alias(metric)
                                    )
                        dfs[alg] = df
                        all_seeds.update(df["seed"].unique())

                # Sort dfs so base algorithm first, then freeze variants by descending freeze steps
                dfs = dict(
                    sorted(
                        dfs.items(),
                        key=lambda x: (
                            0 if "_Freeze_" not in x[0] else 1,
                            -get_freeze_num(x[0]),
                        ),
                    )
                )

                if not dfs:
                    print(f"No data found for {env}-{aperture} {base_alg}")
                    continue

                seeds = sorted(all_seeds)
                n_seeds = len(seeds)
                n_metrics = len(METRICS_TO_PLOT)

                if n_seeds == 0 or n_metrics == 0:
                    continue

                # Create subplot grid: metrics x seeds, sharey for rows, sharex for all columns
                fig, axes = plt.subplots(
                    n_metrics,
                    n_seeds,
                    squeeze=False,
                    layout="constrained",
                    sharey="row",
                    sharex=True,
                )

                # Plot each metric x seed combination
                for i, metric in enumerate(METRICS_TO_PLOT):
                    for j, seed_val in enumerate(seeds):
                        ax = axes[i, j]
                        plot_metric_seed_grid(ax, dfs, metric, seed_val, env, aperture)

                # Set row labels (metric names) on the leftmost column
                for i, metric in enumerate(METRICS_TO_PLOT):
                    axes[i, 0].set_ylabel(metric.replace("_", " ").title(), fontsize=12)

                # Set column titles (seed numbers) on the top row
                for j, seed_val in enumerate(seeds):
                    axes[0, j].set_title(f"Seed {seed_val}", fontsize=12)

                # Set x-label only on the bottom row
                for j in range(n_seeds):
                    axes[-1, j].set_xlabel("Time", fontsize=10)

                # Add legend for algorithms
                handles, labels = axes[0, 0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper right", fontsize=10)

                # Overall title
                fig.suptitle(f"{env} - FOV {aperture} - {base_alg}", fontsize=14)

                # Save the plot
                path = os.path.sep.join(
                    os.path.relpath(__file__).split(os.path.sep)[:-1]
                )
                plot_name = f"{env}-{aperture}-{base_alg}"

                save(
                    save_path=f"{path}/plots/metric_seed_grids",
                    plot_name=plot_name,
                    save_type="pdf",
                    f=fig,
                    width=n_seeds,
                    height_ratio=(n_metrics / n_seeds) * (2 / 3) if n_seeds > 0 else 1,
                )
                plt.close(fig)

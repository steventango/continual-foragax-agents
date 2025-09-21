import os
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


METRICS_TO_PLOT = [
    "ewm_reward",
    "loss",
    "squared_td_error",
    "abs_td_error",
    "weight_change",
    "occupancy_combined",
]

ENV_MAP = {"ForagaxTwoBiomeSmall": "ForagaxTwoBiomeSmall-v2"}


def plot_metric_seed_grid(ax, df, metric, seed_val, alg, env, aperture):
    """Plot a single metric for a single seed on the given axis."""
    if df is None or df.height == 0:
        ax.set_visible(False)
        return

    seed_df = df.filter(pl.col("seed") == seed_val)
    if seed_df.height == 0:
        ax.set_visible(False)
        return

    seed_df = seed_df.sort("frame")

    if metric == "occupancy_combined":
        # Plot all three occupancy metrics on the same axis
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
        if metric not in seed_df.columns:
            ax.set_visible(False)
            return
        frames = seed_df["frame"]
        values = seed_df[metric]
        ax.plot(frames, values, linewidth=1.0, color=colorset.blue)
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

        all_algs = set()
        for sub_results in aperture_results.values():
            for alg_result in sub_results:
                alg = alg_result.filename
                if "DRQN" in alg or "taper" in alg:
                    continue
                all_algs.add(alg)

        algs = sorted(list(all_algs))

        if not algs:
            continue

        for aperture in apertures:
            sub_results = aperture_results.get(aperture, [])

            for alg_result in sorted(sub_results, key=lambda x: x.filename):
                alg = alg_result.filename
                if not alg.startswith("DQN_Freeze") and alg != "DQN":
                    continue

                print(f"Processing {env}-{aperture} {alg}")

                df = alg_result.load(sample=500)
                if df is None or df.height == 0:
                    print(f"No data found for {env}-{aperture} {alg}")
                    continue

                # Get unique seeds
                seeds = sorted(df["seed"].unique())
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
                        plot_metric_seed_grid(
                            ax, df, metric, seed_val, alg, env, aperture
                        )

                # Set row labels (metric names) on the leftmost column
                for i, metric in enumerate(METRICS_TO_PLOT):
                    axes[i, 0].set_ylabel(metric.replace("_", " ").title(), fontsize=12)

                # Set column titles (seed numbers) on the top row
                for j, seed_val in enumerate(seeds):
                    axes[0, j].set_title(f"Seed {seed_val}", fontsize=12)

                # Set x-label only on the bottom row
                for j in range(n_seeds):
                    axes[-1, j].set_xlabel("Time", fontsize=10)

                # Overall title
                fig.suptitle(f"{env} - FOV {aperture} - {alg}", fontsize=14)

                # Save the plot
                path = os.path.sep.join(
                    os.path.relpath(__file__).split(os.path.sep)[:-1]
                )
                plot_name = f"{env}-{aperture}-{alg}"

                save(
                    save_path=f"{path}/plots/metric_seed_grids",
                    plot_name=plot_name,
                    save_type="pdf",
                    f=fig,
                    width=n_seeds,
                    height_ratio=(n_metrics / n_seeds) * (2 / 3) if n_seeds > 0 else 1,
                )
                plt.close(fig)

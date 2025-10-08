import os
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.constants import BIOME_DEFINITIONS, ENV_MAP, TWO_BIOME_COLORS
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)


def plot_biome_occupancy_on_ax(ax, df, biomes, alg, env, aperture):
    biome_names = list(biomes.keys()) + ["Neither"]

    if df is None or df.height == 0:
        print(f"No biome data for {env}-{aperture} {alg}")
        return

    for _, seed_df in df.group_by("seed"):
        seed_df = seed_df.sort("frame")
        frames = seed_df["frame"]
        for biome_name in biome_names:
            occupancy_col = f"{biome_name}_occupancy"
            if occupancy_col in seed_df.columns:
                occupancy = seed_df[occupancy_col]
                color = TWO_BIOME_COLORS.get(biome_name)
                ax.plot(frames, occupancy, color=color, alpha=0.2, linewidth=0.2)

    # Aggregate and plot mean
    agg_cols = [f"{name}_occupancy" for name in biome_names]
    agg_df = (
        df.group_by("frame")
        .agg([pl.mean(col).alias(f"{col}_mean") for col in agg_cols])
        .sort("frame")
    )

    for biome_name in biome_names:
        mean_col = f"{biome_name}_occupancy_mean"
        if mean_col in agg_df.columns:
            frames = agg_df["frame"]
            means = agg_df[mean_col]
            color = TWO_BIOME_COLORS.get(biome_name)
            ax.plot(frames, means, color=color, linewidth=1.0)

    ax.set_xlabel("Time")
    ax.set_ylabel("Biome Occupancy (EMA)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


if __name__ == "__main__":
    results = ResultCollection(
        Model=ExperimentModel,
        metrics=[
            "pos",
            "biome",
            "Morel_occupancy",
            "Oyster_occupancy",
            "Neither_occupancy",
        ],
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
    for env_aperture, sub_results in results.groupby_directory(level=3):
        env, aperture = env_aperture.rsplit("-", 1)
        env = ENV_MAP.get(env, env)
        aperture = int(aperture)

        if env not in env_groups:
            env_groups[env] = {}
        env_groups[env][aperture] = sub_results

    for env, aperture_results in env_groups.items():
        if env not in BIOME_DEFINITIONS:
            print(f"Skipping {env} as no biome definition found")
            continue

        biomes = BIOME_DEFINITIONS[env]
        biome_names = list(biomes.keys()) + ["Neither"]

        apertures = sorted(aperture_results.keys())

        all_algs = set()
        for sub_results in aperture_results.values():
            for alg_result in sub_results:
                alg = alg_result.filename
                if "DRQN" in alg or "taper" in alg:
                    continue
                all_algs.add(alg)

        algs = sorted(list(all_algs))
        alg_map = {name: i for i, name in enumerate(algs)}

        if not algs:
            continue

        fig, axes = plt.subplots(
            len(apertures),
            len(algs),
            squeeze=False,
            layout="constrained",
        )
        for ax in axes.flatten():
            ax.set_frame_on(False)
            ax.set_axis_off()

        for i, aperture in enumerate(apertures):
            axes[i, 0].set_ylabel(
                f"FOV {aperture}",
                rotation=0,
                labelpad=40,
                verticalalignment="center",
            )
            sub_results = aperture_results.get(aperture, [])
            for alg_result in sorted(sub_results, key=lambda x: x.filename):
                alg = alg_result.filename
                if "DRQN" in alg or "taper" in alg:
                    continue
                j = alg_map[alg]
                ax = axes[i, j]

                print(f"{env}-{aperture} {alg}")

                if i == 0:
                    ax.set_title(alg)

                df = alg_result.load(sample=500)
                if df is None or df.height == 0:
                    print(f"No data found for {env}-{aperture} {alg}")
                    continue

                ax.set_frame_on(True)
                ax.set_axis_on()

                plot_biome_occupancy_on_ax(ax, df, biomes, alg, env, aperture)

        handles, labels = [], []
        for name, color in TWO_BIOME_COLORS.items():
            handles.append(Rectangle((0, 0), 1, 1, color=color))
            labels.append(name)

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
        )

        fig.suptitle(env)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        plot_name = f"biome_over_time-{env}"

        nrows = len(apertures)
        ncols = len(algs)

        save(
            save_path=f"{path}/plots/biome_occupancy_over_time",
            plot_name=plot_name,
            save_type="pdf",
            f=fig,
            width=ncols * 1.5,
            height_ratio=(nrows / ncols) * (2 / 3) if ncols > 0 else 1,
        )
        plt.close(fig)

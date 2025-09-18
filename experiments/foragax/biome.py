import os
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tol_colors as tc
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

colorset = tc.colorsets["high_contrast"]

BIOME_COLORS = {
    "Morel": colorset.blue,
    "Oyster": colorset.red,
    "Neither": colorset.yellow,
}

ENV_MAP = {
    "ForagaxTwoBiomeSmall": "ForagaxTwoBiomeSmall-v2"
}

BIOME_DEFINITIONS = {
    "ForagaxTwoBiomeSmall-v2": {
        "Morel": ((3, 3), (6, 6)),
        "Oyster": ((11, 3), (14, 6)),
    }
}

def get_biome(pos, biomes):
    x, y = pos
    for name, ((x1, y1), (x2, y2)) in biomes.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return name
    return "Neither"

if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel, metrics=["pos"])
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
                df = alg_result.load(sample=10_000, sample_type="random")
                if df is None:
                    print(f"No data found for {env}-{aperture} {alg}")
                    continue
                ax.set_frame_on(True)
                ax.set_axis_on()

                all_biome_data = []
                for seed, group_df in df.group_by("seed"):
                    if "pos" not in group_df.columns:
                        continue

                    pos_arrays = group_df["pos"].to_list()
                    if not pos_arrays:
                        continue

                    # Assuming each element in pos_arrays is a (steps, 2) array
                    for pos_array in pos_arrays:
                        pos_array = np.array(pos_array).reshape(-1, 2)

                        biome_visits = [get_biome(pos, biomes) for pos in pos_array]

                        # Count occurrences of each biome
                        biome_counts = {
                            name: biome_visits.count(name) for name in biome_names
                        }
                        total_visits = len(biome_visits)

                        if total_visits > 0:
                            biome_percentages = {
                                name: count / total_visits
                                for name, count in biome_counts.items()
                            }
                            biome_percentages["seed"] = (
                                seed[0] if isinstance(seed, tuple) else seed
                            )
                            all_biome_data.append(biome_percentages)

                biome_df = pl.DataFrame(all_biome_data)

                avg_percentages = (
                    biome_df.select(pl.col(biome_names)).mean().to_dicts()[0]
                )

                labels = list(avg_percentages.keys())
                sizes = list(avg_percentages.values())
                colors = [BIOME_COLORS.get(label) for label in labels]

                ax.pie(
                    sizes,
                    labels=None,
                    autopct="%1f%%",
                    startangle=90,
                    colors=colors,
                    textprops={"fontsize": 14},
                )
                ax.axis("equal")

        handles, labels = [], []
        for name, color in BIOME_COLORS.items():
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
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
        plot_name = f"biome-pie-{env}"

        nrows = len(apertures)
        ncols = len(algs)

        save(
            save_path=f"{path}/plots/biome_occupancy",
            plot_name=plot_name,
            save_type="pdf",
            f=fig,
            width=ncols,
            height_ratio=(nrows / ncols) * (2 / 3) if ncols > 0 else 1,
        )
        plt.close(fig)

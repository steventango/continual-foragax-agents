import os
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.constants import BIOME_COLORS, BIOME_DEFINITIONS, ENV_MAP
from utils.metrics import calculate_biome_occupancy
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)


def plot_biome_barcode_on_ax(ax, df, biomes, seed, env, aperture):
    biome_names = list(biomes.keys()) + ["Neither"]

    if df is None or df.height == 0:
        print(f"No biome data for {env}-{aperture} seed {seed}")
        return

    seed_df = df.filter(pl.col("seed") == seed).sort("frame")
    occupancy_cols = [f"{name}_occupancy" for name in biome_names]

    biome_indices = []
    for row in seed_df.iter_rows(named=True):
        occupancies = [row[col] for col in occupancy_cols]
        max_val = max(occupancies)
        if max_val < 0.5:  # Threshold for "Neither"
            biome_idx = len(biome_names) - 1
        else:
            biome_idx = occupancies.index(max_val)
        biome_indices.append(biome_idx)

    biome_indices = np.array(biome_indices).reshape(1, -1)
    colors = [BIOME_COLORS.get(name, 'gray') for name in biome_names]
    cmap = ListedColormap(colors)

    ax.imshow(biome_indices, cmap=cmap, aspect='auto', origin='lower', extent=[0, len(biome_indices[0]), 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("Time Steps (Last 1k)")
    ax.set_ylabel(f"FOV {aperture}")
    ax.spines[["top", "right", "left"]].set_visible(False)


if __name__ == "__main__":
    results = ResultCollection(
        Model=ExperimentModel,
        metrics=["ewm_reward"],
    )
    results.paths = [path for path in results.paths if "hypers" not in path]
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

        # Only plot for apertures 9, 11, 13, and 15
        selected_apertures = [9, 11, 13, 15]
        apertures = [a for a in selected_apertures if a in aperture_results]

        if not apertures:
            continue

        algs = ["DQN", "Search-Oracle"]

        # Get seeds from DQN data
        dqn_result = None
        for alg_result in sorted(aperture_results[apertures[0]], key=lambda x: x.filename):
            if alg_result.filename == "DQN":
                dqn_result = alg_result
                break
        if dqn_result is None:
            continue
        df_temp = dqn_result.load(start=9999000)
        df_temp = calculate_biome_occupancy(df_temp)
        if df_temp is None or df_temp.height == 0:
            continue
        seeds = sorted(df_temp["seed"].unique())[:3]
        seed_map = {seed: i for i, seed in enumerate(seeds)}

        for alg in algs:
            apertures_for_alg = [a for a in apertures if any(ar.filename == alg for ar in aperture_results[a])]
            if not apertures_for_alg:
                continue

            fig, axes = plt.subplots(
                len(apertures_for_alg),
                len(seeds),
                squeeze=False,
                layout="constrained",
            )
            for ax in axes.flatten():
                ax.set_frame_on(False)
                ax.set_axis_off()

            for i, aperture in enumerate(apertures_for_alg):
                sub_results = aperture_results.get(aperture, [])
                alg_result = next((ar for ar in sorted(sub_results, key=lambda x: x.filename) if ar.filename == alg), None)
                if alg_result is None:
                    continue
                print(f"{env}-{aperture} {alg}")

                # Load data starting from 9.999M for last 1k steps
                df = alg_result.load(start=9999000)
                df = calculate_biome_occupancy(df)
                if df is None or df.height == 0:
                    print(f"No data found for {env}-{aperture} {alg}")
                    continue

                for seed in seeds:
                    j = seed_map[seed]
                    ax = axes[i, j]

                    if i == 0:
                        ax.set_title(f"Seed {seed}")

                    ax.set_frame_on(True)
                    ax.set_axis_on()

                    plot_biome_barcode_on_ax(ax, df, biomes, seed, env, aperture)

            handles, labels = [], []
            for name, color in BIOME_COLORS.items():
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

            fig.suptitle(f"{env} - {alg} Biome Barcode (Last 1k Steps)")

            path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
            plot_name = f"{alg.lower().replace('-', '_')}_biome_barcode_last_1k-{env}"

            nrows = len(apertures_for_alg)
            ncols = len(seeds)

            save(
                save_path=f"{path}/plots/biome_barcode",
                plot_name=plot_name,
                save_type="pdf",
                f=fig,
                width=ncols * 2.0,
                height_ratio=(nrows / ncols) * (1 / 3) if ncols > 0 else 1,
            )
            plt.close(fig)

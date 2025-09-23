import os
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Rectangle
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.constants import BIOME_COLORS, BIOME_DEFINITIONS, ENV_MAP
from utils.metrics import calculate_biome_occupancy
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)


def plot_biome_occupancy_on_ax(ax, df, biomes, seed, env, aperture, alg):
    biome_names = list(biomes.keys()) + ["Neither"]

    if df is None or df.height == 0:
        print(f"No biome data for {env}-{aperture} seed {seed}")
        return

    seed_df = df.filter(pl.col("seed") == seed).sort("frame")
    frames = seed_df["frame"]
    for biome_name in biome_names:
        occupancy_col = f"{biome_name}_occupancy"
        if occupancy_col in seed_df.columns:
            occupancy = seed_df[occupancy_col]
            color = BIOME_COLORS.get(biome_name)
            ax.plot(frames, occupancy, color=color, linewidth=1.0)

    ax.set_xlabel("Time")
    ax.set_ylabel("Biome Occupancy (EMA)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


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
        df_temp = dqn_result.load(start=9900000)
        df_temp = calculate_biome_occupancy(df_temp)
        if df_temp is None or df_temp.height == 0:
            continue
        seeds = sorted(df_temp["seed"].unique())[:5]
        seed_map = {seed: i for i, seed in enumerate(seeds)}

        # Collect valid algorithm-aperture combinations
        valid_combinations = []
        for aperture in apertures:
            sub_results = aperture_results.get(aperture, [])
            for alg_result in sorted(sub_results, key=lambda x: x.filename):
                alg = alg_result.filename
                if alg not in algs:
                    continue
                # Load data to check if it exists
                df = alg_result.load(start=9999000)
                df = calculate_biome_occupancy(df)
                if df is not None and df.height > 0:
                    valid_combinations.append((aperture, alg))

        # Sort combinations by aperture, then by algorithm
        valid_combinations.sort(key=lambda x: (x[0], algs.index(x[1])))

        print(f"Found {len(valid_combinations)} valid algorithm-aperture combinations")

        fig, axes = plt.subplots(
            len(valid_combinations),
            len(seeds),
            squeeze=False,
            layout="constrained",
        )
        for ax in axes.flatten():
            ax.set_frame_on(False)
            ax.set_axis_off()

        for row_idx, (aperture, alg) in enumerate(valid_combinations):
            sub_results = aperture_results.get(aperture, [])
            alg_result = next((r for r in sub_results if r.filename == alg), None)
            if alg_result is None:
                continue
            print(f"{env}-{aperture} {alg}")

            # Load data starting from 9.9M for last 1k steps
            df = alg_result.load(start=9999000)
            df = calculate_biome_occupancy(df)
            if df is None or df.height == 0:
                print(f"No data found for {env}-{aperture} {alg}")
                continue

            for seed in seeds:
                j = seed_map[seed]
                ax = axes[row_idx, j]

                if row_idx == 0:
                    ax.set_title(f"Seed {seed}")

                ax.set_frame_on(True)
                ax.set_axis_on()

                plot_biome_occupancy_on_ax(ax, df, biomes, seed, env, aperture, alg)

                # Set row labels
                if j == 0:
                    ax.set_ylabel(f"{alg}\nFOV {aperture}")

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

        fig.suptitle(f"{env} - Biome Occupancy by Algorithm-FOV Pair and Seed (Last 100k Steps)")

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        plot_name = f"biome_last_100k-{env}"

        nrows = len(valid_combinations)
        ncols = len(seeds)

        save(
            save_path=f"{path}/plots/biome_occupancy_over_time",
            plot_name=plot_name,
            save_type="pdf",
            f=fig,
            width=ncols * 1.5,
            height_ratio=(nrows / ncols) * (2 / 3) if ncols > 0 else 1,
        )
        plt.close(fig)

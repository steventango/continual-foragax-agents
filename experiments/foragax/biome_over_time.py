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

def plot_biome_occupancy_over_time(df, biomes, alg, env, aperture):
    biome_names = list(biomes.keys()) + ["Neither"]
    all_biome_data = []

    for seed, group_df in df.group_by("seed"):
        if "pos" not in group_df.columns:
            continue

        group_df = group_df.sort("frame")
        pos_arrays = group_df["pos"].to_numpy()
        if not len(pos_arrays):
            continue

        biome_visits = [get_biome(pos, biomes) for pos in pos_arrays]

        biome_df = pl.DataFrame({
            "frame": group_df["frame"],
            "biome": biome_visits,
            "seed": seed[0] if isinstance(seed, tuple) else seed,
        })

        for name in biome_names:
            biome_df = biome_df.with_columns(
                (pl.col("biome") == name).cast(pl.Float32).ewm_mean(alpha=1e-3, adjust=True).alias(f"{name}_occupancy")
            )

        all_biome_data.append(biome_df)

    if not all_biome_data:
        print(f"No biome data for {env}-{aperture} {alg}")
        return

    full_df = pl.concat(all_biome_data)

    # downsample for plotting
    full_df = full_df.gather_every(max(1, full_df.height // 1000))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # one legend entry per biome
    for biome_name in biome_names:
        color = BIOME_COLORS.get(biome_name)
        ax.plot([], [], label=biome_name, color=color)

    for seed_val, seed_df in full_df.group_by("seed"):
        seed_df = seed_df.sort("frame")
        frames = seed_df["frame"]
        for biome_name in biome_names:
            occupancy_col = f"{biome_name}_occupancy"
            if occupancy_col in seed_df.columns:
                occupancy = seed_df[occupancy_col]
                color = BIOME_COLORS.get(biome_name)
                ax.plot(frames, occupancy, color=color, alpha=0.2)

    # Aggregate and plot mean
    agg_cols = [f"{name}_occupancy" for name in biome_names]
    agg_df = full_df.group_by("frame").agg(
        [pl.mean(col).alias(f"{col}_mean") for col in agg_cols]
    ).sort("frame")

    for biome_name in biome_names:
        mean_col = f"{biome_name}_occupancy_mean"
        if mean_col in agg_df.columns:
            frames = agg_df["frame"]
            means = agg_df[mean_col]
            color = BIOME_COLORS.get(biome_name)
            ax.plot(frames, means, color=color, linewidth=2.5)


    ax.set_xlabel("Time")
    ax.set_ylabel("Biome Occupancy (EMA)")
    ax.set_title(f"{alg} ({env}, {aperture})")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)

    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    plot_name = f"biome_over_time-{env}-{aperture}-{alg}"
    save(
        save_path=f"{path}/plots/biome_occupancy_over_time",
        plot_name=plot_name,
        save_type="pdf",
        f=fig,
        height_ratio=2/3.
    )
    plt.close(fig)


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

    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=2), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture = env_aperture.rsplit("-", 1)
        env = ENV_MAP.get(env, env)
        aperture = int(aperture)

        if env not in BIOME_DEFINITIONS:
            print(f"Skipping {env} as no biome definition found")
            continue

        biomes = BIOME_DEFINITIONS[env]

        for alg_result in sorted(sub_results, key=lambda x: x.filename):
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")

            df = alg_result.load(sample=10_000, sample_type="random")
            if df is None or df.height == 0:
                continue

            plot_biome_occupancy_over_time(df, biomes, alg, env, aperture)

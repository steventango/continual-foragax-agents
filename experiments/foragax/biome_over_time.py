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

colorset = tc.colorsets["muted"]

COLORS = {
    3: colorset.rose,
    5: colorset.indigo,
    7: colorset.sand,
    9: colorset.cyan,
    11: colorset.teal,
    13: colorset.olive,
    15: colorset.purple,
    "Search-Oracle": colorset.wine,
    "Search-Nearest": colorset.green,
    "Search-Oyster": tc.colorsets["light"].pear,
    "Random": "black",
}

SINGLE = {
    "Random",
    "Search-Nearest",
    "Search-Oracle",
    "Search-Oyster",
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


    # Aggregate data
    agg_cols = [f"{name}_occupancy" for name in biome_names]
    agg_df = full_df.group_by(["frame"]).agg(
        [pl.mean(col).alias(f"{col}_mean") for col in agg_cols] +
        [pl.std(col).alias(f"{col}_std") for col in agg_cols] +
        [pl.count().alias("count")]
    ).sort("frame")

    agg_df = agg_df.with_columns(
        [(pl.col(f"{col}_std") / pl.col("count").sqrt()).alias(f"{col}_sem") for col in agg_cols]
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for biome_name in biome_names:
        mean_col = f"{biome_name}_occupancy_mean"
        sem_col = f"{biome_name}_occupancy_sem"

        if mean_col in agg_df.columns:
            frames = agg_df["frame"]
            means = agg_df[mean_col]
            sems = agg_df[sem_col]

            line = ax.plot(frames, means, label=biome_name)
            color = line[0].get_color()
            ax.fill_between(frames, means - sems, means + sems, color=color, alpha=0.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Biome Occupancy (EMA)")
    ax.set_title(f"Biome Occupancy Over Time: {alg} ({env}, aperture {aperture})")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1)


    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    plot_name = f"biome_over_time-{env}-{aperture}-{alg}"
    save(
        save_path=f"{path}/plots/biome_occupancy_over_time",
        plot_name=plot_name,
        save_type="pdf",
        f=fig,
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

            df = alg_result.load(sample=1_000_000)
            if df is None or df.height == 0:
                continue

            plot_biome_occupancy_over_time(df, biomes, alg, env, aperture)

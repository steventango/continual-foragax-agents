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
        biome_names = list(biomes.keys()) + ["Neither"]

        for alg_result in sorted(sub_results, key=lambda x: x.filename):
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")

            df = alg_result.load(sample=1_000_000)
            if df is None:
                continue

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

                    last_idx = int(0.9 * len(pos_array))
                    pos_array = pos_array[last_idx:]

                    biome_visits = [get_biome(pos, biomes) for pos in pos_array]

                    # Count occurrences of each biome
                    biome_counts = {name: biome_visits.count(name) for name in biome_names}
                    total_visits = len(biome_visits)

                    if total_visits > 0:
                        biome_percentages = {name: count / total_visits for name, count in biome_counts.items()}
                        biome_percentages["seed"] = seed[0] if isinstance(seed, tuple) else seed
                        all_biome_data.append(biome_percentages)

            if not all_biome_data:
                print(f"No biome data for {env_aperture} {alg}")
                continue

            biome_df = pl.DataFrame(all_biome_data)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            avg_percentages = biome_df.select(pl.col(biome_names)).mean().to_dicts()[0]

            labels = list(avg_percentages.keys())
            sizes = list(avg_percentages.values())

            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            ax.set_title(f"Biome Occupancy: {alg} ({env}, aperture {aperture})")

            path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
            plot_name = f"biome-pie-{env}-{aperture}-{alg}"
            save(
                save_path=f"{path}/plots/biome_occupancy",
                plot_name=plot_name,
                save_type="pdf",
                f=fig,
            )
            plt.close(fig)

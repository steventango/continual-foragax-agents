import json
import os
import sys

sys.path.append(os.getcwd() + "/src")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from environments.Foragax import Foragax
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

def get_background(env_name):
    env = Foragax(
        seed=0,
        env_id=env_name,
        aperture_size=1,
    )
    env.start()

    frame = env.env.render(env.state.state, None, render_mode="world")
    return np.asarray(frame, dtype=np.uint8)


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

        env_parts = env.split("-")
        grid_size_str = env_parts[-1]

        background = get_background(env)
        grid_w, grid_h = background.shape[1] // 24, background.shape[0] // 24

        for alg_result in sorted(sub_results, key=lambda x: x.filename):
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")

            df = alg_result.load(sample=1000000)
            if df is None:
                continue

            # Aggregate position data from all seeds
            all_pos = []
            for _, group_df in df.group_by("seed"):
                # group_df is a DataFrame for a single seed
                # The 'pos' column contains numpy arrays of shape (n, 2)
                # We concatenate them all.
                if "pos" not in group_df.columns:
                    continue
                pos_arrays = group_df["pos"].to_list()
                if pos_arrays:
                    all_pos.extend(pos_arrays)

            if not all_pos:
                print(f"No position data for {env_aperture} {alg}")
                continue

            pos_data = np.stack(all_pos)
            x = pos_data[:, 0]
            y = pos_data[:, 1]

            occupancy_map, xedges, yedges = np.histogram2d(
                x,
                y,
                bins=(grid_w, grid_h),
            )

            fig, axes = plt.subplots(2, 1, figsize=(8, 8))
            ax1, ax2 = axes

            # Plot environment on the first subplot
            ax1.imshow(
                background,
                extent=(0, grid_w, 0, grid_h),
                origin="lower",
                aspect="equal",
            )
            ax1.set_title(f"{env}")
            ax1.invert_yaxis()
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Plot occupancy map on the second subplot
            im = ax2.imshow(
                occupancy_map.T,
                origin="lower",
                cmap="viridis",
                extent=(0, grid_w, 0, grid_h),
                aspect="equal",
            )
            ax2.set_title(f"{alg}-{aperture}")
            ax2.invert_yaxis()
            ax2.set_xticks([])
            ax2.set_yticks([])

            fig.tight_layout()
            fig.colorbar(
                im, ax=axes.ravel().tolist(), label="Visitation Count", shrink=0.5
            )

            path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
            plot_name = f"{env}-{aperture}-{alg}"
            save(
                save_path=f"{path}/plots/heatmaps",
                plot_name=plot_name,
                save_type="pdf",
                f=fig,
            )
            plt.close(fig)

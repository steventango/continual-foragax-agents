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

    env_groups = {}
    for env_aperture, sub_results in results.groupby_directory(level=2):
        env, aperture = env_aperture.rsplit("-", 1)
        env = ENV_MAP.get(env, env)
        aperture = int(aperture)

        if env not in env_groups:
            env_groups[env] = {}
        env_groups[env][aperture] = sub_results

    for env, aperture_results in env_groups.items():
        background = get_background(env)
        grid_w, grid_h = background.shape[1] // 24, background.shape[0] // 24

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
            figsize=(len(algs) * 4, len(apertures) * 4),
        )
        for ax in axes.flatten():
            ax.set_frame_on(False)
            ax.set_axis_off()

        max_visitation = 0
        occupancy_maps = {}

        for i, aperture in enumerate(apertures):
            sub_results = aperture_results.get(aperture, [])
            for alg_result in sorted(sub_results, key=lambda x: x.filename):
                alg = alg_result.filename
                if "DRQN" in alg or "taper" in alg:
                    continue

                print(f"{env}-{aperture} {alg}")

                df = alg_result.load(sample=10_000, sample_type="random")
                if df is None:
                    print(f"No data found for {env}-{aperture} {alg}")
                    continue

                all_pos = []
                for _, group_df in df.group_by("seed"):
                    if "pos" not in group_df.columns:
                        continue
                    pos_arrays = group_df["pos"].to_list()
                    if len(pos_arrays) == 0:
                        continue
                    all_pos.extend(pos_arrays)

                if not all_pos:
                    print(f"No position data for {env}-{aperture} {alg}")
                    continue

                pos_data = np.stack(all_pos)
                x = pos_data[:, 0]
                y = pos_data[:, 1]

                occupancy_map, _, _ = np.histogram2d(
                    x,
                    y,
                    bins=(grid_w, grid_h),
                    range=[[0, grid_w], [0, grid_h]],
                )
                max_visitation = max(max_visitation, occupancy_map.max())
                occupancy_maps[(aperture, alg)] = occupancy_map

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

                if i == 0:
                    ax.set_title(alg)

                occupancy_map = occupancy_maps.get((aperture, alg))
                if occupancy_map is None:
                    continue

                ax.set_frame_on(True)
                ax.set_axis_on()
                ax.imshow(
                    background,
                    extent=(0, grid_w, 0, grid_h),
                    origin="lower",
                )
                im = ax.imshow(
                    occupancy_map.T,
                    origin="lower",
                    cmap="inferno",
                    extent=(0, grid_w, 0, grid_h),
                    alpha=0.7,
                    vmin=0,
                    vmax=max_visitation,
                )
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(env)
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            label="Visitation Count",
            shrink=0.6,
            location="bottom",
            anchor=(0.5, -0.5),
        )

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        plot_name = f"heatmap-{env}"

        nrows = len(apertures)
        ncols = len(algs)

        save(
            save_path=f"{path}/plots/heatmaps",
            plot_name=plot_name,
            save_type="pdf",
            f=fig,
            width=ncols * 1.5,
            height_ratio=(nrows / ncols) if ncols > 0 else 1,
        )
        plt.close(fig)

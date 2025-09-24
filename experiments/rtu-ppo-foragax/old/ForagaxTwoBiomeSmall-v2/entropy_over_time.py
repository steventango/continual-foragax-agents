import os
import sys

# sys.path.append(os.getcwd() + "/src")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition
import tol_colors as tc
from experiment.ExperimentModel import ExperimentModel
from utils.constants import ENV_MAP
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

def plot_entropy_on_ax(ax, df, alg, env, aperture):
    if df is None or df.height == 0:
        print(f"No entropy data for {env}-{aperture} {alg}")
        return

    if "entropy" not in df.columns:
        print(f"'entropy' column not found for {env}-{aperture} {alg}")
        return

    # Plot per-seed light traces
    for _, seed_df in df.group_by("seed"):
        seed_df = seed_df.sort("frame")
        frames = seed_df["frame"]
        ent = seed_df["entropy"]
        ax.plot(frames, ent, alpha=0.2, linewidth=0.2)

    # Aggregate and plot mean entropy over time
    agg_df = (
        df.group_by("frame")
        .agg(pl.mean("entropy").alias("entropy_mean"))
        .sort("frame")
    )

    frames = agg_df["frame"]
    means = agg_df["entropy_mean"]
    ax.plot(frames, means, linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Entropy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


if __name__ == "__main__":
    results = ResultCollection(
        Model=ExperimentModel,
        metrics=[
            "entropy",
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

                plot_entropy_on_ax(ax, df, alg, env, aperture)

        fig.suptitle(env)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        plot_name = f"entropy_over_time-{env}"

        nrows = len(apertures)
        ncols = len(algs)

        save(
            save_path=f"{path}/plots/entropy_over_time",
            plot_name=plot_name,
            save_type="pdf",
            f=fig,
            width=ncols * 1.5,
            height_ratio=(nrows / ncols) * (2 / 3) if ncols > 0 else 1,
        )
        plt.close(fig)

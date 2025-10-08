import os
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.constants import ENV_MAP, TWO_BIOME_COLORS
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel, metrics=["pos", "biome"])
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
                df = alg_result.load(sample=1)
                if df is None:
                    print(f"No data found for {env}-{aperture} {alg}")
                    continue
                ax.set_frame_on(True)
                ax.set_axis_on()

                biome_percent_cols = [col for col in df.columns if col.startswith("biome_percent_")]
                avg_percentages = df[biome_percent_cols].mean().to_dict()

                labels = list(avg_percentages.keys())
                labels = [label.replace("biome_percent_", "") for label in labels]
                sizes = list(map(
                    lambda x: x[0],
                    avg_percentages.values()
                ))
                colors = [TWO_BIOME_COLORS.get(label) for label in labels]

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

import os
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

COLORS = {
    3: "#00ffff",
    5: "#3ddcff",
    7: "#57abff",
    9: "#8b8cff",
    11: "#b260ff",
    13: "#d72dff",
    15: "#ff00ff",
    "Random": "black",
}

SINGLE = {
    "Random"
}


if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel)
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    total = 30
    group_size = 2
    fig, axs = plt.subplots(2, total // group_size, sharex=True, sharey=True)

    env = "unknown"
    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=2), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture = env_aperture.split("-", 1)
        aperture = int(aperture)
        if aperture not in {5, 7}:
            continue
        for alg_result in sorted(
            sub_results, key=lambda x: x.filename
        ):
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")

            df = alg_result.load()
            if df is None:
                continue

            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals=hyper_vals,
                metric="ewm_reward",
            )

            xs = np.asarray(xs)
            ys = np.asarray(ys)
            print(ys.shape)
            assert np.all(np.isclose(xs[0], xs))

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=10000,
            )
            if alg not in SINGLE:
                label = f"{alg}-{aperture}"
                color = COLORS[aperture]
            else:
                label = alg
                color = COLORS[label]

            for i, y_group in enumerate(np.array_split(ys, len(ys) // group_size)):
                if alg == "DQN":
                    ax = axs[0, i]
                else:
                    ax = axs[1, i]
                for j, y in enumerate(y_group):
                    if aperture == 5:
                        cmap = plt.colormaps.get_cmap("Blues")
                        color = cmap(0.5 + (j / (group_size * 2)))
                    elif aperture == 7:
                        cmap = plt.colormaps.get_cmap("Oranges")
                        color = cmap(0.5 + (j / (group_size * 2)))

                    if j == len(y_group) - 1:
                        ax.plot(
                            xs[0],
                            y,
                            color=color,
                            linewidth=1.0,
                            label=label,
                        )
                    else:
                        ax.plot(xs[0], y, linewidth=0.5, color=color)

            for ax in axs.ravel():
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            for ax in axs[-1]:
                ax.set_xlabel("Time steps")
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

            for ax in axs[:, 0]:
                ax.set_ylabel("Average Reward")


        axs[0, 0].set_title("DQN (1 hidden layer)")
        axs[1, 0].set_title("DQN (2 hidden layers)")

        axs[0, -1].legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        axs[1, -1].legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f"{path}/plots",
            plot_name=env,
            save_type="pdf",
            f=fig,
            width=15,
            height_ratio=4 / 45,
        )

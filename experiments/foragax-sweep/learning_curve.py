import json
import os
from pathlib import Path
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
    "Search-Nearest": "red",
    "Search-Oracle": "green",
}

SINGLE = {
    "Random",
    "Search-Nearest",
    "Search-Oracle",
}


if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel)
    results.paths = [path for path in results.paths if "hypers" not in path]
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    fig, axs = plt.subplots(3, 1, sharex=True, sharey='all')

    env = "unknown"
    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=2), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture = env_aperture.split("-", 1)
        aperture = int(aperture)
        for alg_result in sorted(
            sub_results, key=lambda x: x.filename
        ):
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")

            exp_path = Path(alg_result.exp_path)
            best_configuration_path = (
                exp_path.parent.parent / "hypers" / exp_path.parent.name / exp_path.name
            )
            if not best_configuration_path.exists():
                continue
            with open(best_configuration_path) as f:
                best_configuration = json.load(f)

            df = alg_result.load_by_params(best_configuration)
            if df is None:
                continue
            df = df.sort("id", "frame")

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
            mask = xs[0] > 1000
            xs = xs[:, mask]
            ys = ys[:, mask]
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

            if alg == "DQN":
                axes = [axs[0]]
            elif alg == "DQN_L2_Init":
                axes = [axs[1]]
            elif alg == "DQN_LN":
                axes = [axs[2]]
            else:
                axes = axs

            for ax in axes:
                ax.plot(
                    xs[0],
                    res.sample_stat,
                    label=label,
                    color=color,
                    linewidth=1.0,
                )
                if len(ys) >= 5:
                    ax.fill_between(
                        xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2
                    )
                else:
                    for y in ys:
                        ax.plot(xs[0], y, color=color, linewidth=0.2)

                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
                ax.set_xlabel("Time steps")
                ax.set_ylabel("Average Reward")
                ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f"{path}/plots",
            plot_name=env,
            save_type="pdf",
            f=fig,
            height_ratio=8/3,
        )

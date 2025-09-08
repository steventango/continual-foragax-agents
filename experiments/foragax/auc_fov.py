import os
import sys
from collections import defaultdict

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
from PyExpPlotting.matplot import save, setDefaultConference
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

setDefaultConference("jmlr")

METRIC = "reward"
LAST_PERCENT = 0.1
COLORS = {
    "DQN": "tab:blue",
    "DQN_L2_Init": "purple",
    "ESARSA": "tab:orange",
    "SoftmaxAC": "tab:green",
    "Random": "black",
}
ORDER = {
    "Random": 0,
    "Greedy": 2,
    "Greedy-122": 1,
}
SPECIAL = {
    "Random",
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

    f, ax = plt.subplots()
    apertures = defaultdict(list)
    auc = defaultdict(list)
    auc_ci_low = defaultdict(list)
    auc_ci_high = defaultdict(list)
    special = {}

    # group by aperture
    for env, sub_results in results.groupby_directory(level=2):
        aperture = int(env.split("-")[-1])
        for alg_result in sub_results:
            alg = alg_result.filename
            df = alg_result.load()
            if df is None:
                continue

            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {
                col: df[col][0] for col in cols
            }

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals=hyper_vals,
                metric="ewm_reward",
            )

            xs = np.asarray(xs)
            ys = np.asarray(ys)

            assert np.all(np.isclose(xs[0], xs))

            last_idx = int((1 - LAST_PERCENT) * xs.shape[1])
            xs = xs[:, last_idx:]
            ys = ys[:, last_idx:]

            # take mean across time
            ys = ys.mean(axis=1, keepdims=True)

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            if alg not in SPECIAL:
                name = alg.split("-")[0]
                apertures[name].append(aperture)
                auc[name].append(res.sample_stat)
                auc_ci_low[name].extend(res.ci[0])
                auc_ci_high[name].extend(res.ci[1])
            else:
                special[alg] = res

    # sort DQN results by aperture
    sorted_apertures = {}
    sorted_auc = {}
    sorted_auc_ci_low = {}
    sorted_auc_ci_high = {}
    for alg in apertures:
        sort_idx = np.argsort(apertures[alg])
        sorted_apertures[alg] = np.array(apertures[alg])[sort_idx]
        sorted_auc[alg] = np.array(auc[alg])[sort_idx]
        sorted_auc_ci_low[alg] = np.array(auc_ci_low[alg])[sort_idx]
        sorted_auc_ci_high[alg] = np.array(auc_ci_high[alg])[sort_idx]

        ax.plot(
            sorted_apertures[alg],
            sorted_auc[alg],
            label=alg,
            color=COLORS[alg],
            linewidth=1,
        )
        ax.fill_between(
            sorted_apertures[alg],
            sorted_auc_ci_low[alg],
            sorted_auc_ci_high[alg],
            color=COLORS[alg],
            alpha=0.2,
        )

    a = np.unique(np.concatenate(list(apertures.values())))

    for (alg, report), color in zip(
        sorted(special.items(), key=lambda x: ORDER[x[0]]),
        [
            COLORS["Random"],
        ],
        strict=True,
    ):
        if alg == "Greedy":
            alg = "Search Oracle"
        if alg == "Greedy-122":
            alg = "Search Nearest"
        ax.plot(
            a,
            [report.sample_stat] * len(a),
            label=alg,
            color=color,
            linewidth=1,
        )
        ax.fill_between(
            a,
            np.repeat(report.ci[0], len(a)),
            np.repeat(report.ci[1], len(a)),
            color=color,
            alpha=0.4,
        )

    ax.set_xlabel("Field of View")
    ax.set_ylabel("Last 10% Average Reward AUC")
    ax.set_xticks(a)
    ax.set_xticklabels([str(int(x)) for x in a])

    ax.legend(ncol=2, frameon=False)
    # ax.text(3, 1.3, "Search Oracle", color=COLORS["EQRC"])
    # right side
    # ax.text(15, 1.2, "DQN", color=COLORS["DQN"], ha="right")
    # ax.text(15, 0.95, "Search Nearest", color=COLORS["ESARSA"], ha="right")
    # ax.text(15, 0.4, "Random", color=COLORS["SoftmaxAC"], ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name="auc_fov",
        save_type="pdf",
        f=f,
    )

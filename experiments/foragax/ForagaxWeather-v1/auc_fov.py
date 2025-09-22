import os
import sys
from collections import defaultdict

from utils.constants import LABEL_MAP
from utils.plotting import label_lines

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
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

METRIC = "reward"
LAST_PERCENT = 0.1
colorset = tc.colorsets["muted"]

COLORS = {
    "DQN": colorset.indigo,
    "DQN_L2_Init": colorset.purple,
    "DQN_LN": colorset.teal,
    "DQN_Shrink_and_Perturb": colorset.rose,
    "DQN_Hare_and_Tortoise": colorset.sand,
    "DQN_Reset_Head": colorset.olive,
    "Search-Oracle": colorset.wine,
    "Search-Nearest": colorset.green,
    "Search-Oyster": tc.colorsets["light"].pear,
    "Random": "black",
}

DEFAULT_COLOR = colorset.pale_grey
ORDER = {
    "Search-Oracle": 0,
    "Search-Nearest": 1,
    "Search-Oyster": 2,
    "Random": 3,
}
SPECIAL = {
    "Random",
    "Search-Nearest",
    "Search-Oracle",
    "Search-Oyster",
}

if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
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

    legend_handles = {}

    # group by aperture
    for env, sub_results in results.groupby_directory(level=3):
        aperture = int(env.rsplit("-", 1)[-1])
        for alg_result in sub_results:
            alg = alg_result.filename
            print(f"{env} {alg}")
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
                apertures[alg].append(aperture)
                auc[alg].append(res.sample_stat)
                auc_ci_low[alg].extend(res.ci[0])
                auc_ci_high[alg].extend(res.ci[1])
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

        parts = alg.split("_")
        label = LABEL_MAP.get(alg, alg)
        linestyle = "-"
        color = COLORS.get(alg, DEFAULT_COLOR)

        (line,) = ax.plot(
            sorted_apertures[alg],
            sorted_auc[alg],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1,
        )
        fill = ax.fill_between(
            sorted_apertures[alg],
            sorted_auc_ci_low[alg],
            sorted_auc_ci_high[alg],
            color=color,
            alpha=0.2,
        )
        legend_handles[label] = (fill, line)

    a = np.unique(np.concatenate(list(apertures.values())))

    for alg, report in sorted(special.items(), key=lambda x: ORDER[x[0]]):
        color = COLORS.get(alg, DEFAULT_COLOR)
        (line,) = ax.plot(
            a,
            [report.sample_stat] * len(a),
            label=alg,
            color=color,
            linewidth=1,
        )
        fill = ax.fill_between(
            a,
            np.repeat(report.ci[0], len(a)),
            np.repeat(report.ci[1], len(a)),
            color=color,
            alpha=0.4,
        )
        legend_handles[alg] = (fill, line)

    ax.set_xlabel("Field of View")
    ax.set_ylabel("Last 10% Average Reward AUC")
    ax.set_xticks(a)
    ax.set_xticklabels([str(int(x)) for x in a])

    label_lines(ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name="auc_fov",
        save_type="pdf",
        f=f,
        width=1.5,
    )

import os
import sys

sys.path.append(os.getcwd() + "/src")
import matplotlib.pyplot as plt
import numpy as np
import tol_colors as tc
from matplotlib.lines import Line2D
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)

from experiment.ExperimentModel import ExperimentModel
from utils.constants import LABEL_MAP
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

colorset = tc.colorsets["muted"]

COLORS = {
    # 3: colorset.rose,
    # 5: colorset.indigo,
    # 7: colorset.sand,
    # 9: colorset.cyan,
    # 11: colorset.teal,
    # 13: colorset.olive,
    # 15: colorset.purple,
    100000: colorset.rose,
    1000000: colorset.indigo,
    5000000: colorset.sand,
    -1: colorset.cyan,
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

    nalgs = 1
    ncols = int(np.ceil(np.sqrt(nalgs))) if nalgs > 3 else nalgs
    nrows = int(np.ceil(nalgs / ncols)) if nalgs > 3 else 1
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey="all", layout="constrained", squeeze=False)
    axs = axs.flatten()
    env = "unknown"
    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=2), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture = env_aperture.rsplit("-", 1)
        aperture = int(aperture)
        if aperture != 3 and aperture != 15:
            continue

        for alg_result in sorted(sub_results, key=lambda x: x.filename):
            alg = alg_result.filename
            if aperture == 3:
                if alg not in {
                    # "DQN",
                    "DQN_small_buffer",
                    # "DQN_L2_Init",
                    # "DQN_L2_Init_small_buffer",
                    # "DQN_Freeze_100k",
                    "DQN_Freeze_100k_small_buffer",
                    # "DQN_Freeze_1M",
                    "DQN_Freeze_1M_small_buffer",
                    # "DQN_Freeze_5M",
                    "DQN_Freeze_5M_small_buffer",
                    # "DQN_L2_Init_Freeze_100k",
                    # "DQN_L2_Init_Freeze_100k_small_buffer",
                    # "DQN_L2_Init_Freeze_1M",
                    # "DQN_L2_Init_Freeze_1M_small_buffer",
                    # "DQN_L2_Init_Freeze_5M",
                    # "DQN_L2_Init_Freeze_5M_small_buffer",
                }:
                    continue
            else:
                if alg not in SINGLE:
                    continue
            print(f"{env_aperture} {alg}")

            freeze_step = -1
            if "Freeze_100k" in alg:
                freeze_step = 100000
            elif "Freeze_1M" in alg:
                freeze_step = 1000000
            elif "Freeze_5M" in alg:
                freeze_step = 5000000

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
                alg_label = LABEL_MAP.get(alg, alg)
                label = None
                color = COLORS[freeze_step]
            else:
                alg_label = alg
                label = alg
                color = COLORS[label]

            ax_idxs = []
            # if alg == "DQN":
            #     ax_idxs = [0]
            # elif alg == "DQN_Freeze_100k":
            #     ax_idxs = [0]
            # elif alg == "DQN_Freeze_1M":
            #     ax_idxs = [0]
            # elif alg == "DQN_Freeze_5M":
            #     ax_idxs = [0]
            # elif alg == "DQN_L2_Init":
            #     ax_idxs = [1]
            # elif alg == "DQN_L2_Init_Freeze_100k":
            #     ax_idxs = [1]
            # elif alg == "DQN_L2_Init_Freeze_1M":
            #     ax_idxs = [1]
            # elif alg == "DQN_L2_Init_Freeze_5M":
            #     ax_idxs = [1]
            if alg == "DQN_small_buffer":
                ax_idxs = [0]
            elif alg == "DQN_Freeze_100k_small_buffer":
                ax_idxs = [0]
            elif alg == "DQN_Freeze_1M_small_buffer":
                ax_idxs = [0]
            elif alg == "DQN_Freeze_5M_small_buffer":
                ax_idxs = [0]
            # elif alg == "DQN_L2_Init_small_buffer":
            #     ax_idxs = [3]
            # elif alg == "DQN_L2_Init_Freeze_100k_small_buffer":
            #     ax_idxs = [3]
            # elif alg == "DQN_L2_Init_Freeze_1M_small_buffer":
            #     ax_idxs = [3]
            # elif alg == "DQN_L2_Init_Freeze_5M_small_buffer":
            #     ax_idxs = [3]
            else:
                ax_idxs = np.arange(len(axs))

            for i in ax_idxs:
                ax = axs[i]
                ax.plot(
                    xs[0],
                    res.sample_stat,
                    label=label,
                    color=color,
                    linewidth=1.0,
                )
                if alg not in SINGLE:
                    ax.set_title(alg_label)
                if len(ys) >= 5:
                    ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)
                else:
                    for y in ys:
                        ax.plot(xs[0], y, color=color, linewidth=0.2)

                ax.ticklabel_format(
                    axis="x", style="sci", scilimits=(0, 0), useMathText=True
                )
                if i % ncols == 0:
                    ax.set_ylabel("Average Reward")
                if i // ncols == nrows - 1:
                    ax.set_xlabel("Time steps")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

    for ax in axs:
        if not ax.get_lines():
            ax.set_visible(False)
            continue

    legend_elements = []
    color_keys = sorted([k for k in COLORS.keys() if isinstance(k, int)])
    for color_key in color_keys:
        label = f"Frozen @ {color_key}"
        if color_key == -1:
            label = "Not Frozen"
        legend_elements.append(Line2D([0], [0], color=COLORS[color_key], lw=2, label=label))

    for k in SINGLE:
        if k in COLORS:
            legend_elements.append(Line2D([0], [0], color=COLORS[k], lw=2, label=k))

    fig.legend(handles=legend_elements, loc="outside center right", frameon=False)

    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name=env,
        save_type="pdf",
        f=fig,
        width=4/3,
        height_ratio=1/2,
        # width=ncols,
        # height_ratio=(nrows / ncols) * (2 / 3),
    )

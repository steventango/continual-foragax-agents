import argparse
import os
import sys
from collections import defaultdict

# sys.path.append(os.getcwd() + "/src")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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
    3: colorset.rose,
    5: colorset.indigo,
    7: colorset.sand,
    9: colorset.cyan,
    11: colorset.teal,
    13: colorset.olive,
    15: colorset.purple,
    "Search-Oracle": colorset.wine,
    "Search-Nearest": colorset.green,
    "Random": "black",
}

SINGLE = {
    "Random",
    "Search-Nearest",
    "Search-Oracle",
    "Search-Oyster",
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true", help="Normalize rewards")
    parser.add_argument(
        "--apertures",
        nargs="+",
        type=int,
        default=[9],
        help="List of apertures to plot",
    )
    args = parser.parse_args()

    NORMALIZE = args.normalize
    ylabel = "Normalized Reward" if args.normalize else "Average Reward"

    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    data = defaultdict(dict)

    # Pre-load baselines from aperture 15
    baseline_data = {}
    for env_aperture_temp, sub_results_temp in sorted(
        results.groupby_directory(level=3), key=lambda x: int(x[0].split("-")[-1])
    ):
        if env_aperture_temp.endswith("-15"):
            for alg_result in sub_results_temp:
                if alg_result.filename in SINGLE:
                    df = alg_result.load(end=9_980_000)
                    if df is not None:
                        cols = set(dd.hyper_cols).intersection(df.columns)
                        hyper_vals = {col: df[col][0] for col in cols}

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

                        baseline_data[alg_result.filename] = (xs, ys)
            break

    env = "unknown"
    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=3), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture = env_aperture.rsplit("-", 1)
        aperture = int(aperture)

        # Collect all ys for this env_aperture
        alg_ys = {}
        alg_xs = {}
        for alg_result in sorted(sub_results, key=lambda x: x.filename):
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")
            if alg not in SINGLE and aperture not in args.apertures:
                continue

            df = alg_result.load(end=9_980_000)
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

            alg_xs[alg] = xs
            alg_ys[alg] = ys

        # Add pre-loaded baselines
        for baseline, (xs, ys) in baseline_data.items():
            alg_ys[baseline] = ys
            alg_xs[baseline] = xs

        # Normalize if requested
        if args.normalize:
            if "Search-Oracle" in baseline_data:
                baseline_ys = baseline_data["Search-Oracle"][1]
                for alg in alg_ys:
                    alg_ys[alg] = alg_ys[alg] / baseline_ys

        data[int(aperture)] = {alg: (alg_xs[alg], alg_ys[alg]) for alg in alg_ys}

    unique_apertures = sorted(data.keys())
    if args.apertures:
        unique_apertures = [a for a in unique_apertures if a in args.apertures]
    unique_algs = sorted(
        set(alg for d in data.values() for alg in d.keys() if alg not in SINGLE)
    )
    nrows = len(unique_apertures)
    ncols = len(unique_algs)
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, layout="constrained", squeeze=False
    )

    for i, aperture in enumerate(unique_apertures):
        for j, alg in enumerate(unique_algs):
            ax = axs[i, j]
            if alg not in SINGLE and alg in data[aperture]:
                xs, ys = data[aperture][alg]
                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                    iterations=10000,
                )
                color = COLORS[aperture]
                ax.plot(
                    xs[0],
                    res.sample_stat,
                    color=color,
                    linewidth=1.0,
                )
                if len(ys) >= 5:
                    ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)
                else:
                    for y in ys:
                        ax.plot(xs[0], y, color=color, linewidth=0.2)

                ax.ticklabel_format(
                    axis="x", style="sci", scilimits=(0, 0), useMathText=True
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # Plot baselines on all subplots
            for baseline in SINGLE:
                if baseline in data[aperture]:
                    xs_b, ys_b = data[aperture][baseline]
                    res_b = curve_percentile_bootstrap_ci(
                        rng=np.random.default_rng(0),
                        y=ys_b,
                        statistic=Statistic.mean,
                        iterations=10000,
                    )
                    color_b = COLORS[baseline]
                    linestyle = "--" if baseline != alg else "-"
                    ax.plot(
                        xs_b[0],
                        res_b.sample_stat,
                        color=color_b,
                        linewidth=1.0,
                        linestyle=linestyle,
                    )
                    if len(ys_b) >= 5:
                        ax.fill_between(
                            xs_b[0], res_b.ci[0], res_b.ci[1], color=color_b, alpha=0.2
                        )
                    else:
                        for y in ys_b:
                            ax.plot(
                                xs_b[0],
                                y,
                                color=color_b,
                                linewidth=0.2,
                                linestyle=linestyle,
                            )

                    ax.ticklabel_format(
                        axis="x", style="sci", scilimits=(0, 0), useMathText=True
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            if i == 0:
                ax.set_title(LABEL_MAP.get(alg, alg))
            if j == 0:
                ax.set_ylabel(f"FOV {aperture}")
            if i == nrows - 1:
                ax.set_xlabel("Time steps")

    legend_elements = []
    aperture_keys = sorted(
        [k for k in COLORS.keys() if isinstance(k, int) and k in unique_apertures]
    )
    for ap in aperture_keys:
        legend_elements.append(Line2D([0], [0], color=COLORS[ap], lw=2, label=f"FOV {ap}"))

    for k in SINGLE:
        if k in COLORS:
            legend_elements.append(
                Line2D([0], [0], color=COLORS[k], lw=2, linestyle="--", label=k)
            )

    fig.legend(handles=legend_elements, loc="outside center right", frameon=False)

    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name=f"{env}_normalized" if NORMALIZE else env,
        save_type="pdf",
        f=fig,
        width=ncols,
        height_ratio=(nrows / ncols) * (2 / 3),
    )
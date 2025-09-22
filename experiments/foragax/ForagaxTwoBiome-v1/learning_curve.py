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
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

colorset = tc.colorsets["muted"]

COLORS = {
    "Random": "black",
    "Search-Nearest": colorset.green,
    "Search-Brown-Avoid-Green": colorset.sand,
    "Search-Brown": colorset.olive,
    "Search-Oyster": colorset.rose,
    "Search-Morel": colorset.purple,
    "Search-Morel-Avoid-Green": colorset.teal,
    "Search-Oracle": colorset.wine,
}

SINGLE = {
    "Random",
    "Search-Nearest",
    "Search-Oracle",
    "Search-Oyster",
}

AGENTS_TO_PLOT = [
    "Random",
    "Search-Brown-Avoid-Green",
    "Search-Brown",
    "Search-Morel-Avoid-Green",
    "Search-Morel",
    "Search-Nearest",
    "Search-Oracle",
    "Search-Oyster",
]


if __name__ == "__main__":
    ylabel = "Average Reward"

    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    nalgs = len(AGENTS_TO_PLOT)
    ncols = 1
    nrows = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey="all", layout="constrained")
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
            if alg not in AGENTS_TO_PLOT:
                continue
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
            mask = xs[0] > 1000
            xs = xs[:, mask]
            ys = ys[:, mask]
            print(ys.shape)
            assert np.all(np.isclose(xs[0], xs))

            alg_xs[alg] = xs
            alg_ys[alg] = ys

        # Now plot
        for alg in alg_ys:
            ys = alg_ys[alg]
            xs = alg_xs[alg]

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=10000,
            )
            label = alg
            color = COLORS.get(alg, 'black')

            ax.plot(
                xs[0],
                res.sample_stat,
                label=label,
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
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time steps")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        legend_elements = [Line2D([0], [0], color=COLORS[alg], lw=2, label=alg) for alg in AGENTS_TO_PLOT if alg in COLORS]
        fig.legend(handles=legend_elements, loc="outside center right", frameon=False)

    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name=env,
        save_type="pdf",
        f=fig,
        width=2,
        height_ratio=(nrows / ncols) * (2 / 3),
    )

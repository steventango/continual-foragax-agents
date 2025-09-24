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

# Shared color palette (Okabe–Ito + a few extras); reused across subplots
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#332288",  # dark purple
    "#44AA99",  # teal
]

# Linestyles to distinguish families
LINESTYLES = {
    "RealTimeActorCriticMLP": "-",   # solid
    "ActorCriticMLP":        "--",  # dashed
    "RealTimeActorCriticMLPDefault": "-",
    "ActorCriticMLPDefault":        "--",
    "Random": ":",                   # dotted baseline
}

# LINESTYLES = {
#     "RealTimeActorCriticConv-3": "-",
#     "RealTimeActorCriticConvEmb-3": "--",
#     "RealTimeActorCriticConvEmbNE-3": "-.",
#     "RealTimeActorCriticConvNE-3": ":",
#     "RealTimeActorCriticConvPooling-3": (0, (3, 1, 1, 1)),
#     "RealTimeActorCriticConvPoolingNE-3": (0, (5, 1)),
#     "RealTimeActorCriticMLP-3": (0, (1, 2)),
#     "RealTimeActorCriticMLPNE-3": (0, (3, 5, 1, 5)),
#     "Random": (0, (1, 1)),
# }

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

    # Collect sub-results grouped by FOV (aperture) so we can build a grid
    by_aperture = {}
    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=3), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture_str = env_aperture.rsplit("-", 1)
        aperture = int(aperture_str)
        by_aperture.setdefault((env, aperture), []).extend(sub_results)

    # Determine unique apertures for this env (assumes a single env; if multiple, we create per-env figures)
    # Group by env first
    env_to_apertures = {}
    for (env, aperture), subs in by_aperture.items():
        env_to_apertures.setdefault(env, []).append(aperture)

    for env, apertures in env_to_apertures.items():
        apertures = sorted(set(apertures))
        n = len(apertures)
        ncols = 3 if n >= 3 else n
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, squeeze=False)

        # For each aperture, pick a subplot and plot all algorithms within it
        for idx, aperture in enumerate(apertures):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]

            # baselines
            ax.axhline(y=1.4, label="Search Morel", color="#8c564b")
            ax.axhline(y=1.7, label="Search Oracle", color="#17becf")

            # Build color mapping for this subplot from the shared palette, rotated by aperture index
            # Collect algorithms present for this (env, aperture)
            sub_results = by_aperture.get((env, aperture), [])
            algs = []
            for alg_result in sub_results:
                alg = alg_result.filename
                if alg in SINGLE:
                    label = alg
                else:
                    label = f"{alg}-{aperture}"
                algs.append((alg, label))

            # Stable order by algorithm name for deterministic color assignment
            algs = sorted(algs, key=lambda x: x[0])
            # Create a rotated palette start so colors differ per subplot
            offset = idx % len(PALETTE)
            palette_cycle = PALETTE[offset:] + PALETTE[:offset]
            # Map label -> color (Random stays gray)
            label_to_color = {}
            color_i = 0
            for alg, label in algs:
                if label == "Random":
                    label_to_color[label] = "#999999"
                else:
                    label_to_color[label] = palette_cycle[color_i % len(palette_cycle)]
                    color_i += 1

            # Plot each algorithm's curve(s)
            for alg_result in sub_results:
                alg = alg_result.filename
                if alg in SINGLE:
                    label = alg
                else:
                    label = f"{alg}-{aperture}"

                df = alg_result.load()
                if df is None:
                    continue

                cols = set(dd.hyper_cols).intersection(df.columns)
                hyper_vals = {col: df[col][0] for col in cols}

                xs, ys = extract_learning_curves(
                    df,
                    hyper_vals=hyper_vals,
                    metric="ewm_reward",
                )

                xs = np.asarray(xs)
                ys = np.asarray(ys)
                assert np.all(np.isclose(xs[0], xs))

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                    iterations=10000,
                )

                color = label_to_color.get(label, "#444444")
                linestyle = LINESTYLES.get(alg, "-") if label != "Random" else LINESTYLES["Random"]

                ax.plot(
                    xs[0],
                    res.sample_stat,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                )
                if len(ys) >= 5:
                    ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)
                else:
                    for y in ys:
                        ax.plot(xs[0], y, color=color, linestyle=linestyle, linewidth=0.2)

            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
            ax.set_xlabel("Time steps")
            ax.set_ylabel("Average Reward")
            ax.set_title(f"{env} — FOV {aperture}")
            ax.set_ylim(1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

        # Hide any unused subplots
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis('off')

        # Save one figure per env containing the grid
        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f"{path}/plots",
            plot_name=f"{env}_grid",
            save_type="pdf",
            f=fig,
            width = 6,
            height_ratio=2 / 3,
        )

        plt.close(fig)

import os
import sys
import tol_colors as tc
# sys.path.append(os.getcwd() + "/src")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
import re
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

PALETTE = [
    colorset.rose,
    colorset.indigo,
    # colorset.sand,
    # colorset.cyan,
    colorset.teal,
    colorset.olive,
    colorset.purple,
    colorset.wine,
    colorset.green,
]

# Linestyles to distinguish families
LINESTYLES = {
    "RealTimeActorCriticMLP": "-",
    "ActorCriticMLP":        "-", 
    "Random": ":",                
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

# Helper: strip optional "5M" token (with or without leading separator) so
# color is shared between 5M and non-5M variants

def base_without_5m(name: str) -> str:
    return re.sub(r"[-_]?5M", "", name)

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
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharey=True)

        # For each aperture, pick a subplot and plot all algorithms within it
        for idx, aperture in enumerate(apertures):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]

            # # baselines
            # ax.axhline(y=1.4, label="Search Morel", color="#8c564b")
            # ax.axhline(y=1.7, label="Search Oracle", color="#17becf")

            # Build color mapping for this subplot from the shared palette, rotated by aperture index
            # Collect algorithms present for this (env, aperture)
            sub_results = by_aperture.get((env, aperture), [])
            algs = []
            for alg_result in sub_results:
                alg = alg_result.filename
                # Exclude any agent with "world" in its name
                if "world" in alg.lower():
                    continue
                if alg in SINGLE:
                    label = alg
                else:
                    label = f"{alg}-{aperture}"

                # Rename labels for clarity
                if label.startswith("RealTimeActorCriticMLP"):
                    label = label.replace("RealTimeActorCriticMLP", "RTU-PPO")
                elif label.startswith("ActorCriticMLP"):
                    label = label.replace("ActorCriticMLP", "PPO")

                algs.append((alg, label))

            # Stable order by algorithm name for deterministic color assignment
            algs = sorted(algs, key=lambda x: x[0])
            # Create a rotated palette start so colors differ per subplot
            offset = idx % len(PALETTE)
            palette_cycle = PALETTE[offset:] + PALETTE[:offset]
            # Map base_label -> color; ensure 5M and non-5M share color
            label_to_color = {}
            color_i = 0
            for alg, label in algs:
                # Random stays gray and does not consume from the palette
                if label == "Random":
                    label_to_color[label] = "#999999"
                    continue

                base_alg = base_without_5m(alg)
                base_label = base_alg if alg in SINGLE else f"{base_alg}-{aperture}"

                if base_label not in label_to_color:
                    label_to_color[base_label] = palette_cycle[color_i % len(palette_cycle)]
                    color_i += 1

            # Plot each algorithm's curve(s)
            for alg_result in sub_results:
                alg = alg_result.filename
                # Exclude any agent with "world" in its name
                if "world" in alg.lower():
                    continue
                if alg in SINGLE:
                    label = alg
                else:
                    label = f"{alg}-{aperture}"

                # Rename labels for clarity
                if label.startswith("RealTimeActorCriticMLP"):
                    label = label.replace("RealTimeActorCriticMLP", "RTU-PPO")
                elif label.startswith("ActorCriticMLP"):
                    label = label.replace("ActorCriticMLP", "PPO")

                df = alg_result.load(end=10_000_000)
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

                # Use the shared color for 5M/non-5M variants; dash style for 5M
                base_alg = base_without_5m(alg)
                base_label = base_alg if alg in SINGLE else f"{base_alg}-{aperture}"
                color = label_to_color.get(base_label, "#444444")

                if label == "Random":
                    linestyle = LINESTYLES["Random"]
                else:
                    linestyle = "--" if "5M" in alg else "-"

                ax.plot(
                    xs[0],
                    res.sample_stat,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                )
                if len(ys) >= 5:
                    ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.1)
                else:
                    for y in ys:
                        ax.plot(xs[0], y, color=color, linestyle=linestyle, linewidth=0.2)

            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
            ax.set_xlabel("Time steps")
            ax.set_ylabel("Average Reward")
            ax.set_title(f"{env} — FOV {aperture}")
            # ax.set_ylim(1.0)
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
            width = 2,
            height_ratio=2 / 3,
        )

        plt.close(fig)

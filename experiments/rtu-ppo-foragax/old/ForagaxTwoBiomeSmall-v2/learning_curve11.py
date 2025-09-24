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

COLORS = {
    "RealTimeActorCriticConv-11": "#1f77b4",        # Muted blue
    "RealTimeActorCriticConvEmb-11": "#2ca02c",     # Muted green
    "RealTimeActorCriticConvEmbNE-11": "#d62728",   # Muted red
    "RealTimeActorCriticConvNE-11": "#9467bd",      # Muted purple
    "RealTimeActorCriticConvPooling-11": "#ff7f0e", # Muted orange
    "RealTimeActorCriticConvPoolingNE-11": "#8c564b", # Muted brown
    "RealTimeActorCriticMLP-11": "#17becf",         # Muted teal
    "RealTimeActorCriticMLPNE-11": "#e377c2",       # Muted pink
    "Random": "#7f7f7f",                           # Gray
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

    fig, ax = plt.subplots(1, 1)
    
    ax.axhline(y=1.2, label="Oracle", color="black")  # Muted purple
    # ax.axhline(y=0.8, label="DQN-3-7", color="#ff7f0e")  # Muted orange
    # ax.axhline(y=1.0, label="Search Nearest", color="#17becf")  # Muted teal

    env = "unknown"
    for env_aperture, sub_results in sorted(
        results.groupby_directory(level=2), key=lambda x: int(x[0].split("-")[-1])
    ):
        env, aperture = env_aperture.rsplit("-", 1)

        aperture = int(aperture)
        for alg_result in sub_results:
            alg = alg_result.filename
            print(f"{env_aperture} {alg}")

            if alg not in SINGLE:
                label = f"{alg}-{aperture}"
            else:
                label = alg
            if label not in COLORS:
                continue
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

            
            ax.plot(
                xs[0],
                res.sample_stat,
                label=label,
                color=COLORS[label],
                # linestyle=LINESTYLES[label],
                linewidth=1.0,
            )
            if len(ys) >= 5:
                ax.fill_between(
                    xs[0], res.ci[0], res.ci[1], color=COLORS[label], alpha=0.2
                )
            else:
                for y in ys:
                    ax.plot(xs[0], y, color=COLORS[label], 
                            # linestyle=LINESTYLES[label], 
                            linewidth=0.2)

        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Average Reward")
        ax.set_title(env)
        ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f"{path}/plots",
            plot_name=env,
            save_type="pdf",
            f=fig,
            height_ratio=2 / 3,
        )

import json
import os
import sys
from pathlib import Path

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

colorset = tc.colorsets["medium_contrast"]

COLORS = {
    7: colorset[1],
    "Search-Brown-Avoid-Green": colorset[2],
    "Search-Brown": colorset[3],
    "Search-Morel-Avoid-Green": colorset[4],
    "Search-Morel": colorset[5],
    "Search-Oracle": colorset[6],
    "Search-Nearest": tc.colorsets["vibrant"].teal,
    "Search-Oyster": tc.colorsets["vibrant"].orange,
    "Random": colorset[7],
}

SINGLE = {
    "Random",
    "Search-Brown-Avoid-Green",
    "Search-Brown",
    "Search-Morel-Avoid-Green",
    "Search-Morel",
    "Search-Nearest",
    "Search-Oracle",
    "Search-Oyster",
}


if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
    results.paths = [path for path in results.paths if "hypers" not in path]
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    # Collect unique algorithm bases and buffer sizes
    unique_alg_bases = set()
    unique_buffers = set()
    for aperture_or_baseline, sub_results in sorted(
        results.groupby_directory(level=4),
        key=lambda x: (
            0 if x[0].isdigit() else 1,
            int(x[0].rsplit("-", 1)[-1]) if x[0].isdigit() else 0,
        ),
    ):
        if aperture_or_baseline.isdigit():
            for alg_result in sub_results:
                alg = alg_result.filename
                if "_B" in alg:
                    parts = alg.split("_B")
                    alg_base = parts[0]
                    buffer = int(parts[1])
                    unique_alg_bases.add(alg_base)
                    unique_buffers.add(buffer)
    unique_alg_bases = sorted(unique_alg_bases)
    unique_buffers = sorted(unique_buffers)

    ncols = len(unique_alg_bases)
    nrows = len(unique_buffers)
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey="all", layout="constrained"
    )
    env = "unknown"
    for aperture_or_baseline, sub_results in sorted(
        results.groupby_directory(level=4),
        key=lambda x: (
            0 if x[0].isdigit() else 1,
            int(x[0].rsplit("-", 1)[-1]) if x[0].isdigit() else 0,
        ),
    ):
        aperture = None
        if aperture_or_baseline.isdigit():
            aperture = int(aperture_or_baseline)

        for alg_result in sorted(sub_results, key=lambda x: x.filename):
            alg = alg_result.filename
            print(f"{aperture_or_baseline} {alg}")

            exp_path = Path(alg_result.exp_path)
            env = exp_path.parent.parent
            best_configuration_path = (
                env / "hypers" / exp_path.parent.name / exp_path.name
            )
            env = env.name
            if not best_configuration_path.exists():
                continue
            with open(best_configuration_path) as f:
                best_configuration = json.load(f)

            df = alg_result.load_by_params(best_configuration)
            if df is None:
                continue
            df = df.sort("id", "frame")

            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}  # type: ignore

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,  # type: ignore
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
            ax = axs[0]
            if aperture:
                alg_base = alg.split("_B")[0]
                buffer = int(alg.split("_B")[1])
                row = unique_buffers.index(buffer)
                col = unique_alg_bases.index(alg_base)
                ax = axs[row, col]
                alg_label = LABEL_MAP.get(alg_base, alg_base)
                color = COLORS[aperture]
            else:
                alg_label = alg
                color = COLORS[alg]

            # Plot
            if aperture:
                # Plot on specific ax
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
            else:
                # Plot on all axs
                for ax in axs.flatten():
                    ax.plot(
                        xs[0],
                        res.sample_stat,
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

    # Set titles and formatting
    for i, ax in enumerate(axs.flatten()):
        alg_base = unique_alg_bases[i % ncols]
        alg_label = LABEL_MAP.get(alg_base, alg_base)
        title = f"{alg_label}\n(Buffer Size {unique_buffers[i // ncols]})"
        ax.set_title(title)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        if i % ncols == 0:
            ax.set_ylabel("Average Reward")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Time steps")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axs.flatten():
        if not ax.get_lines():
            ax.set_visible(False)
            continue

    legend_elements = []
    aperture_keys = sorted([k for k in COLORS.keys() if isinstance(k, int)])
    for ap in aperture_keys:
        legend_elements.append(Line2D([0], [0], color=COLORS[ap], lw=2, label=f"FOV {ap}"))

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
        width=ncols,
        height_ratio=(nrows / ncols) * (2 / 3),
    )

import json
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from utils.plotting import select_colors
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

if __name__ == "__main__":
    # Collect all items that need colors
    all_color_keys = set()

    # Pre-scan to collect all algorithms and apertures that need colors
    results_temp = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
    results_temp.paths = [path for path in results_temp.paths if "hypers" not in path]

    for _, sub_results in results_temp.groupby_directory(level=4):
        for alg_result in sub_results:
            alg = alg_result.filename
            if "_B" in alg:
                alg_base = alg.split("_B")[0]
                all_color_keys.add(alg_base)
            else:
                all_color_keys.add(alg)

    n_colors = len(all_color_keys)
    color_list = select_colors(n_colors)

    sorted_keys = sorted([k for k in all_color_keys if isinstance(k, str)])

    COLORS = dict(zip(sorted_keys, color_list, strict=True))

    SINGLE = {k for k in all_color_keys if isinstance(k, str)}

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
                    buffer = int(parts[1].split("_")[0])
                    unique_alg_bases.add(alg_base)
                    unique_buffers.add(buffer)
    unique_alg_bases = sorted(unique_alg_bases)
    unique_buffers = sorted(unique_buffers)

    ncols = len(unique_alg_bases)
    nrows = len(unique_buffers)
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey="all", layout="constrained", squeeze=False
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
            env = exp_path.parent.parent.name
            df = alg_result.load()
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
                color = COLORS[alg_base]
            else:
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
    for alg in SINGLE:
        if alg in COLORS:
            alg_label = LABEL_MAP.get(alg, alg)
            legend_elements.append(Line2D([0], [0], color=COLORS[alg], lw=2, label=alg_label))

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

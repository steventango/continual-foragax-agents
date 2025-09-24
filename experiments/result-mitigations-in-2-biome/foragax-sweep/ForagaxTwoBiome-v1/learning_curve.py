import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.append(os.getcwd() + "/src")
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

    # Collect unique algorithm bases, apertures, and buffers
    unique_alg_bases = set()
    unique_apertures = set()
    unique_buffers = set()
    for aperture_or_baseline, sub_results in sorted(
        results.groupby_directory(level=4),
        key=lambda x: (
            0 if x[0].isdigit() else 1,
            int(x[0].rsplit("-", 1)[-1]) if x[0].isdigit() else 0,
        ),
    ):
        if aperture_or_baseline.isdigit():
            aperture = int(aperture_or_baseline)
            unique_apertures.add(aperture)
            for alg_result in sub_results:
                alg = alg_result.filename
                if "_B" in alg:
                    parts = alg.split("_B")
                    alg_base = parts[0]
                    buffer = int(parts[1])
                    unique_alg_bases.add(alg_base)
                    unique_buffers.add(buffer)
    unique_alg_bases = sorted(unique_alg_bases)
    unique_apertures = sorted(unique_apertures)
    unique_buffers = sorted(unique_buffers)

    ncols = len(unique_buffers)
    nrows = len(unique_apertures)
    env = "unknown"

    # Collect data for plotting
    aperture_buffer_data = defaultdict(list)

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
            print(f"{np.mean(res.sample_stat):.3f}")

            if aperture:
                alg_base = alg.split("_B")[0]
                buffer = int(alg.split("_B")[1])
                alg_label = LABEL_MAP.get(alg_base, alg_base)
                aperture_buffer_data[(aperture, buffer)].append(
                    [alg_label, xs, ys, res, None]
                )
            else:
                alg_label = LABEL_MAP.get(alg, alg)
                for ap in unique_apertures:
                    for buf in unique_buffers:
                        aperture_buffer_data[(ap, buf)].append(
                            [alg_label, xs, ys, res, None]
                        )

    # Assign colors based on number of algorithms
    all_algorithms = set()
    for key in aperture_buffer_data:
        for item in aperture_buffer_data[key]:
            all_algorithms.add(item[0])
    n_colors = len(all_algorithms)
    color_list = select_colors(n_colors)

    color_map = dict(zip(sorted(all_algorithms), color_list, strict=True))  # type: ignore

    # Update colors in aperture_buffer_data
    for key in aperture_buffer_data:
        for item in aperture_buffer_data[key]:
            item[4] = color_map[item[0]]

    # Plot the data
    for ap in unique_apertures:
        for buf in unique_buffers:
            fig, ax = plt.subplots(1, 1, layout="constrained")
            for alg_label, xs, ys, res, color in aperture_buffer_data[(ap, buf)]:
                ax.plot(
                    xs[0],
                    res.sample_stat,
                    color=color,
                    linewidth=1.0,
                    label=alg_label,
                )
                if len(ys) >= 5:
                    ax.fill_between(xs[0], res.ci[0], res.ci[1], color=color, alpha=0.2)
                else:
                    for y in ys:
                        ax.plot(xs[0], y, color=color, linewidth=0.2)
            ax.set_title(f"Aperture {ap}, Buffer {buf}")

            # Set formatting
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText=True
            )
            ax.set_ylabel("Average Reward")
            ax.set_xlabel("Time steps")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Create legend
            legend_items = {}
            for alg_label, _, _, _, color in aperture_buffer_data[(ap, buf)]:
                legend_items[alg_label] = color

            legend_elements = [
                Line2D([0], [0], color=color, lw=2, label=label)
                for label, color in legend_items.items()
            ]

            fig.legend(
                handles=legend_elements, loc="outside center right", frameon=False
            )

            path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
            save(
                save_path=f"{path}/plots",
                plot_name=f"{env}_aperture_{ap}_buffer_{buf}",
                save_type="pdf",
                f=fig,
                width=4 / 3,
                height_ratio=1 / 2,
            )

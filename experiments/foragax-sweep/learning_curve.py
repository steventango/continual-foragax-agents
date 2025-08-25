import os
import sys

sys.path.append(os.getcwd() + "/src")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


from PyExpPlotting.matplot import save, setDefaultConference
import rlevaluation.hypers as Hypers
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    TimeSummary,
    extract_learning_curves,
    curve_percentile_bootstrap_ci,
)
from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return

setDefaultConference("jmlr")


COLORS = {
    "DQN": "tab:blue",
    "EQRC": "purple",
    "ESARSA": "tab:orange",
    "SoftmaxAC": "tab:green",
}


if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel)
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    for env, sub_results in results.groupby_directory(level=2):
        fig, ax = plt.subplots(1, 1)
        for alg_result in sub_results:
            alg = alg_result.filename

            df = alg_result.load()
            if df is None:
                continue

            df = (
                df
                .with_columns(
                    pl.col("reward").str.json_decode().cast(pl.List(pl.Float32))
                )
                .explode("reward")
                .with_columns(
                    pl.int_range(0, pl.len()).over("id").alias("frame")
                )
                .with_columns(
                    pl.col("reward")
                    .ewm_mean(alpha=1e-3, adjust=False)
                    .over("id")
                    .alias("ewm_reward")
                )
            )

            report = Hypers.select_best_hypers(
                df,
                metric="ewm_reward",
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.mean,
                statistic=Statistic.mean,
            )
            print(alg)
            Hypers.pretty_print(report)
            print(report.best_configuration)

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals=report.best_configuration,
                metric="ewm_reward",
                interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
            )

            xs = np.asarray(xs)[:, :: exp.total_steps // 1000]
            ys = np.asarray(ys)[:, :: exp.total_steps // 1000]
            assert np.all(np.isclose(xs[0], xs))

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=10000,
            )

            ax.plot(xs[0], res.sample_stat, label=alg, color=COLORS[alg], linewidth=1.0)
            if len(ys) >= 5:
                ax.fill_between(
                    xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2
                )
            else:
                for y in ys:
                    ax.plot(xs[0], y, color=COLORS[alg], linewidth=0.2)

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

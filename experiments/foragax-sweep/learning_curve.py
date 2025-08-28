import json
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import matplotlib.pyplot as plt
import numpy as np
from PyExpPlotting.matplot import save, setDefaultConference
from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

setDefaultConference("jmlr")


COLORS = {
    "DQN": "tab:blue",
    "EQRC": "purple",
    "ESARSA": "tab:orange",
    "Random": "black",
    "SoftmaxAC": "tab:green",
}


if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel)
    results.paths = [path for path in results.paths if "hypers" not in path]
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
            print(f"{env} {alg}")

            exp_path = Path(alg_result.exp_path)
            best_configuration_path = (
                exp_path.parent.parent / "hypers" / exp_path.parent.name / exp_path.name
            )
            if not best_configuration_path.exists():
                continue
            with open(best_configuration_path) as f:
                best_configuration = json.load(f)

            df = alg_result.load_by_params(best_configuration)
            if df is None:
                continue
            df = df.sort("id", "frame")

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals=best_configuration,
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

        path = Path(__file__).relative_to(Path.cwd()).parent
        save(
            save_path=f"{path}/plots",
            plot_name=env,
            save_type="pdf",
            f=fig,
            height_ratio=2 / 3,
        )

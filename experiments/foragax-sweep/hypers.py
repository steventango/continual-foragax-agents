import sys
from pathlib import Path

from experiment.hypers import generate_hyper_sweep_table, update_best_config

sys.path.append(str(Path.cwd() / "src"))

import json

import rlevaluation.hypers as Hypers
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    TimeSummary,
)

from experiment.data import post_process_data
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


def main():
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
    alg_reports = {}
    for alg_result in results:
        alg = alg_result.filename

        df = alg_result.load()
        if df is None:
            continue

        df = post_process_data(df)

        print(alg)
        report = Hypers.select_best_hypers(
            df,
            metric="ewm_reward",
            prefer=Hypers.Preference.high,
            time_summary=TimeSummary.mean,
            statistic=Statistic.mean,
        )
        Hypers.pretty_print(report)
        print(report.best_configuration)
        exp_path = Path(alg_result.exp_path)
        best_configuration_path = (
            exp_path.parent.parent / "hypers" / exp_path.parent.name / exp_path.name
        )
        best_configuration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(
            best_configuration_path,
            "w",
        ) as f:
            json.dump(report.best_configuration, f, indent=4)

        alg_reports[alg] = report

        # update_best_config(alg, report, __file__)

    path = Path(__file__).relative_to(Path.cwd()).parent
    # generate_hyper_sweep_table(alg_reports, __file__)


if __name__ == "__main__":
    main()

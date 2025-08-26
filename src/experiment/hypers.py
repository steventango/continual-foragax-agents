import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rlevaluation.hypers import HyperSelectionResult


def update_best_config(alg: str, report: HyperSelectionResult, exp_path: Path):
    dir_path = exp_path.parent
    sweep_path = next(dir_path.glob(f"**/{alg}.json"))
    path = Path(str(sweep_path).replace("-sweep", ""))
    with open(sweep_path, "r") as f:
        sweep_config = json.load(f)
    if path.exists():
        with open(path, "r") as f:
            config = json.load(f)
    else:
        config = sweep_config.copy()

    for config_param, best_config in report.best_configuration.items():
        if pd.isnull(best_config):
            continue
        parts = config_param.split(".")
        curr = config["metaParameters"]
        curr_sweep = sweep_config["metaParameters"]
        for part in parts[:-1]:
            curr = curr[part]
            curr_sweep = curr_sweep[part]
        if not isinstance(curr_sweep[parts[-1]], list):
            continue
        if isinstance(best_config, np.integer):
            best_config = int(best_config)
        elif isinstance(best_config, np.floating):
            best_config = float(best_config)
        curr[parts[-1]] = best_config

    with open(path, "w") as f:
        json.dump(config, f, indent=4)



hyper_to_pretty_map = {
    "batch": "Minibatch size",
    "buffer_size": "Replay memory size",
    "buffer_min_size": "Minimum replay history",
    "buffer_strategy": "Buffer sampling strategy",
    "epsilon": "\\epsilon-greedy \\epsilon",
    "gamma": "Discount factor \\gamma",
    "optimizer.alpha": "Step size",
    "optimizer.beta1": "Adam $\\beta_1$",
    "optimizer.beta2": "Adam $\\beta_2$",
    "optimizer.eps": "Adam $\\epsilon$",
    "optimizer.name": "Optimizer",
    "target_refresh": "Target network update frequency",
    "update_freq": "Update frequency",
}
drop_non_hypers = [
    "experiment.seed_offset",
    "environment.env_id",
    "representation.type",
    "representation.hidden",
]


def generate_hyper_sweep_table(
    env_reports: dict[str, dict[str, Any]], path: Path
):
    table = {}
    default_table = {}
    for j, (env, alg_reports) in enumerate(env_reports.items()):
        for i, (alg, data) in enumerate(alg_reports.items()):
            result = data["result"]
            report = data["report"]
            sweep_path = result.exp_path
            path = Path(str(sweep_path).replace("-sweep", ""))
            with open(sweep_path, "r") as f:
                sweep_config = json.load(f)
            with open(path, "r") as f:
                config = json.load(f)
            for config_param, best_config in report.best_configuration.items():
                if pd.isnull(best_config):
                    continue
                parts = config_param.split(".")
                curr = config["metaParameters"]
                curr_sweep = sweep_config["metaParameters"]
                for part in parts[:-1]:
                    curr = curr[part]
                    curr_sweep = curr_sweep[part]
                choices = curr_sweep[parts[-1]]
                if not isinstance(choices, list):
                    if config_param not in default_table:
                        default_table[config_param] = {}
                    default_table[config_param][alg] = choices
                    continue
                if config_param not in table:
                    table[config_param] = {}
                table[config_param][alg] = best_config
                if i == len(alg_reports) - 1:
                    table[config_param]["Choices"] = choices

        df = pd.DataFrame(table).T.reset_index()
        algs = [alg.split("-")[-1] for alg in alg_reports]
        df.columns = ["Hyperparameter"] + algs + ["Choices"]
        # sort df by specific order on Hyperparameter
        # order desired: optimizer.alpha, update_freq , target_refresh, optimizer.beta2, optimizer.eps
        df = df.set_index("Hyperparameter")
        df = df.reindex(
            [
                "optimizer.alpha",
                "optimizer.beta2",
                "optimizer.eps",
            ]
        )
        df = df.drop(index=drop_non_hypers, errors="ignore")
        df = df.rename(index=hyper_to_pretty_map)

        df_choices = df[df.columns[-1:]]
        table_choices = (
            df_choices.style.map_index(lambda _: "font-weight: bold;", axis="columns")
            .format(format_choices, escape="latex", na_rep="")
            .to_latex(convert_css=True)
        )

        df_default = pd.DataFrame(default_table).T.reset_index()
        df_default.columns = ["Hyperparameter"] + algs
        df_default = df_default.set_index("Hyperparameter")
        df_default = df_default.drop(index=drop_non_hypers, errors="ignore")
        df_default = df_default.rename(index=hyper_to_pretty_map)
        table_default = (
            df_default.style.map_index(lambda _: "font-weight: bold;", axis="columns")
            .format(format_default, escape="latex", na_rep="")
            .to_latex(convert_css=True)
        )

        df_selected = df[df.columns[:-1]]
        table_selected = (
            df_selected.style.map_index(lambda _: "font-weight: bold;", axis="columns")
            .format(format_default, escape="latex", na_rep="")
            .to_latex(convert_css=True)
        )

    return table_choices, table_default, table_selected


def format_default(s):
    if isinstance(s, str):
        if s == "ADAM":
            return "Adam"
        return s
    string = f"{s:g}"
    string = string.replace("1e-08", "$10^{-8}$")
    string = string.replace("1e-05", "$10^{-5}$")
    string = string.replace("3e-05", r"$3 \times 10^{-5}$")
    string = string.replace("0.0001", "$10^{-4}$")
    string = string.replace("0.0003", r"$3 \times 10^{-4}$")
    string = string.replace("0.001", "$10^{-3}$")
    string = string.replace("0.003", r"$3 \times 10^{-3}$")
    string = string.replace("0.01", "$10^{-2}$")
    return string


def format_choices(s):
    if isinstance(s, list):
        string = json.dumps([format_default(choice) for choice in s])
        string = string.replace("[", "\\{").replace("]", "\\}")
        string = string.replace("$", "")
        string = string.replace('"', "")
        string = "$" + string + "$"
        return string
    return format_default(s)

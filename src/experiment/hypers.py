import glob
import json
import os

import numpy as np
import pandas as pd
from rlevaluation.hypers import HyperSelectionResult


def update_best_config(alg: str, report: HyperSelectionResult, file: str):
    dir_path = os.path.dirname(file)
    sweep_path = glob.glob(f"{dir_path}/**/{alg}.json", recursive=True)[0]
    path = sweep_path.replace("-sweep", "")
    with open(sweep_path, "r") as f:
        sweep_config = json.load(f)
    with open(path, "r") as f:
        config = json.load(f)

    for config_param, best_config in zip(
        report.config_params, report.best_configuration
    ):
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


def generate_hyper_sweep_table(
    alg_reports: dict[str, HyperSelectionResult], file: str
):
    dir_path = os.path.dirname(file)

    table = {}
    for i, (alg, report) in enumerate(alg_reports.items()):
        sweep_path = glob.glob(f"{dir_path}/**/{alg}.json", recursive=True)[0]
        path = sweep_path.replace("-sweep", "")
        with open(sweep_path, "r") as f:
            sweep_config = json.load(f)
        with open(path, "r") as f:
            config = json.load(f)
        for config_param, best_config in zip(
            report.config_params, report.best_configuration
        ):
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
                continue
            if config_param not in table:
                table[config_param] = {}
            table[config_param][alg] = best_config
            if i == len(algs) - 1:
                table[config_param]["Choices"] = choices

    df = pd.DataFrame(table).T.reset_index()
    algs = [alg.split("-")[-1] for alg in algs]
    df.columns = ["Hyperparameter"] + algs + ["Choices"]
    # sort df by specific order on Hyperparameter
    # order desired: optimizer.alpha, update_freq , target_refresh, optimizer.beta2, optimizer.eps
    df = df.set_index("Hyperparameter")
    df = df.reindex(
        [
            "optimizer.alpha",
            "update_freq",
            "target_refresh",
            "optimizer.beta2",
            "optimizer.eps",
        ]
    )
    df = df.rename(
        index={
            "optimizer.alpha": "Step size",
            "update_freq": "Update period",
            "target_refresh": "Target network update frequency",
            "optimizer.beta2": "Adam $\\beta_2$",
            "optimizer.eps": "Adam $\\epsilon$",
        }
    )
    df1 = df[df.columns[:-1]]
    table1 = df1.to_latex(float_format=format_float)
    print(table1)
    df2 = df[df.columns[-1:]]
    table2 = df2.to_latex(formatters={"Choices": format_choice})
    print(table2)


def format_float(number: float):
    string = f"{number:g}"
    string = string.replace("1e-08", "$10^{-8}$")
    string = string.replace("1e-05", "$10^{-5}$")
    string = string.replace("3e-05", r"$3 \times 10^{-5}$")
    string = string.replace("0.0001", "$10^{-4}$")
    string = string.replace("0.0003", r"$3 \times 10^{-4}$")
    string = string.replace("0.001", "$10^{-3}$")
    return string


def format_choice(choices: list):
    string = json.dumps([format_float(choice) for choice in choices])
    string = string.replace("[", "\\{").replace("]", "\\}")
    string = string.replace("$", "")
    string = string.replace('"', "")
    string = "$" + string + "$"
    return string

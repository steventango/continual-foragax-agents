
import sys
from pathlib import Path

from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


def _to_pandas(df):
    """Best-effort conversion to a pandas.DataFrame from various df types."""
    if isinstance(df, pd.DataFrame):
        return df
    # Polars and other libs often expose to_pandas
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass
    # Fallback: try DataFrame constructor
    try:
        return pd.DataFrame(df)
    except Exception:
        raise TypeError("Could not convert result dataframe to pandas.")


def _numeric(x: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(x)
    except Exception:
        return False


def plot_sensitivity(merged: pd.DataFrame, hyper: str, out_path: Path, title_prefix: str):
    stats = (
        merged.groupby(hyper)["mean_ewm_reward"]
        .agg(["mean", "std", "count"])  # type: ignore
        .reset_index()
    )
    # Compute standard error of the mean
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))

    x = stats[hyper]
    y = stats["mean"]
    yerr = stats["sem"].fillna(0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    if _numeric(x):
        # Sort by numeric value
        order = np.argsort(pd.to_numeric(x, errors="coerce"))
        plt.errorbar(x.iloc[order], y.iloc[order], yerr=yerr.iloc[order], fmt="-o")
    else:
        # Treat as categories
        plt.errorbar(range(len(x)), y, yerr=yerr, fmt="-o")
        plt.xticks(range(len(x)), [str(v) for v in x], rotation=45, ha="right")

    plt.title(f"{title_prefix} — sensitivity: {hyper}")
    plt.xlabel(hyper)
    plt.ylabel("mean_ewm_reward (mean ± SEM)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # Collect results across experiments
    results = ResultCollection(Model=ExperimentModel, metrics=["mean_ewm_reward"])
    # Ignore any precomputed hypers folders
    results.paths = [path for path in results.paths if "hypers" not in path]

    # Register columns with rlevaluation for downstream tooling (kept for compatibility)
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    # For each environment and algorithm, aggregate run-level performance and plot
    for env, sub_results in results.groupby_directory(level=4):
        for alg_result in sub_results:
            alg = alg_result.filename
            print(f"{env} {alg}")

            df = alg_result.load()
            if df is None:
                continue
            pdf = _to_pandas(df)

            # Basic required columns
            metric_col = "mean_ewm_reward"
            seed_col = "seed" if "seed" in pdf.columns else None
            frame_col = "frame" if "frame" in pdf.columns else None

            # Attempt to find a run identifier column
            run_candidates = [c for c in ["id", "run", "trial", "experiment_id"] if c in pdf.columns]
            group_cols = []
            if run_candidates:
                group_cols.append(run_candidates[0])
            if seed_col:
                group_cols.append(seed_col)
            if not group_cols:
                # Fallback to using all hyperparameters as an identifier
                group_cols = results.get_hyperparameter_columns()

            # Aggregate over time to get one score per run (mean over frames)
            if frame_col:
                run_scores = (
                    pdf.groupby(group_cols)[metric_col]
                    .mean()
                    .reset_index()
                )
            else:
                # If there's no explicit time column, assume rows are already per-step aggregated
                run_scores = pdf[group_cols + [metric_col]]

            # Attach hyperparameter values (take the first observed value within each run)
            hyper_cols = results.get_hyperparameter_columns()
            if hyper_cols:
                hyp_first = pdf.groupby(group_cols)[hyper_cols].first().reset_index()
                merged = run_scores.merge(hyp_first, on=group_cols, how="left")
            else:
                merged = run_scores.copy()

            # Plot one figure per hyperparameter
            out_dir = Path(alg_result.exp_path).parent / "sensitivity" / env / alg
            title_prefix = f"{env} / {alg}"
            printed_any = False
            for hyper in hyper_cols:
                # Skip degenerate hypers with only one value present
                if merged[hyper].nunique(dropna=True) <= 1:
                    continue
                out_path = out_dir / f"{hyper}.png"
                plot_sensitivity(merged, hyper, out_path, title_prefix)
                print(f"Saved: {out_path}")
                printed_any = True

            if not printed_any:
                print(f"No varying hyperparameters found for {env} {alg}; skipping plots.")


if __name__ == "__main__":
    main()

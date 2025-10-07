import argparse
import concurrent.futures
import os
import sys
from itertools import product
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import polars as pl

from experiment.ExperimentModel import ExperimentModel
from utils.results import Result, ResultCollection


def process_alg_result(alg_result: Result, group, aperture):
    alg = alg_result.filename
    print(f"{group} {alg}")

    exp_path = Path(alg_result.exp_path)
    env = exp_path.parent.parent.name
    targets = [1_000_000, 5_000_000, 10_000_000]
    intervals = [1, 1_000, 10_000, 100_000, 500_000, 1_000_000]
    n_samples = 500

    sample_types = (
        ["every"]
        + [
            (total - interval, total, n_samples)
            for total, interval in product(targets, intervals)
        ]
        + [(total, total + interval, n_samples) for total, interval in product(targets, intervals)]
    )
    print("\n".join(str(sample_type) for sample_type in sample_types))

    df = alg_result.load(sample_type=sample_types)  # type: ignore
    if df is None:
        return None

    df = df.with_columns(
        pl.lit(env).alias("env"),
        pl.lit(group).alias("group"),
        pl.lit(alg).alias("alg"),
        pl.lit(aperture).cast(pl.Int64).alias("aperture"),
    )
    print(df)
    return df


def main(experiment_path: Path):
    output_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments"))
        / "data.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path = experiment_path.resolve()

    results = ResultCollection(
        path=experiment_path,
        Model=ExperimentModel,
    )
    results.paths = [path for path in results.paths if "hypers" not in path]
    print("\n".join([str(path) for path in results.paths]))

    dfs = []
    for group, sub_results in results.groupby_directory(level=4):
        aperture = int(group) if group.isdigit() else None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_alg_result, alg_result, group, aperture)
                for alg_result in sub_results
            ]
            for future in concurrent.futures.as_completed(futures):
                df = future.result()
                if df is not None:
                    dfs.append(df)

    dfs = [df for df in dfs if not df.is_empty()]
    if not dfs:
        print("No data found to concatenate")
        return

    all_df = pl.concat(dfs, how="diagonal_relaxed")
    all_df = all_df.sort(["env", "group", "alg", "id", "frame"])
    all_df.write_parquet(output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process experiment data and save to parquet"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the experiment directory",
        default="experiments/E39-baselines/foragax/ForagaxTwoBiome-v3",
    )
    args = parser.parse_args()
    path = Path(args.path)
    main(path)

import gc

import psutil
from rlevaluation.config import data_definition

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

mem = psutil.virtual_memory()
print(f"Memory usage on start : {mem.percent}%")
results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
dd = data_definition(
    hyper_cols=results.get_hyperparameter_columns(),
    seed_col="seed",
    time_col="frame",
    environment_col=None,
    algorithm_col=None,
    make_global=False,
)

total_rows = 0
for env_aperture, sub_results in sorted(
    results.groupby_directory(level=2), key=lambda x: int(x[0].split("-")[-1])
):
    env, aperture = env_aperture.rsplit("-", 1)
    aperture = int(aperture)

    for alg_result in sorted(sub_results, key=lambda x: x.filename):
        alg = alg_result.filename
        alg = alg_result.filename

        df = alg_result.load()
        if df is None:
            continue

        # Data loaded successfully
        total_rows += len(df)
        print(f"Data for {alg} in {env_aperture}: {len(df)} rows")
        print(f"Total rows so far: {total_rows}")
        mem = psutil.virtual_memory()
        print(f"Memory usage: {mem.percent}%")
        if mem.percent > 30:
            print("Memory leak detected, stopping further processing.")
            exit(1)

        # Free memory
        del df
        gc.collect()

    # After processing all alg_results in the group
    gc.collect()
    del sub_results

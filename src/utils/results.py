import gc
import glob
import importlib
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

import connectorx as cx
import ml_instrumentation._utils.sqlite as sqlu
import numpy as np
import polars as pl
from PyExpUtils.models.ExperimentDescription import (
    ExperimentDescription,
    loadExperiment,
)
from PyExpUtils.results.indices import listIndices
from PyExpUtils.results.tools import getHeader, getParamsAsDict

from utils.metrics import (
    calculate_biome_occupancy,
    calculate_ewm_reward,
    calculate_mean_reward,
    calculate_object_traces,
)
from utils.ml_instrumentation.reader import get_run_ids

Exp = TypeVar("Exp", bound=ExperimentDescription)


def read_metrics_from_data(
    data_path: str | Path,
    metrics: Iterable[str] | None = None,
    run_ids: Iterable[int] | None = None,
    sample: int | None = 500,
    sample_type: str
    | tuple[int, int]
    | tuple[int, int, int]
    | list[str | tuple[int, int] | tuple[int, int, int]] = "every",
    start: int | None = None,
    end: int | None = None,
):
    if run_ids is None:
        run_id_paths = {}
        for path in Path(data_path).glob("*.npz"):
            run_id = int(path.stem)
            run_id_paths[run_id] = path
    else:
        run_id_paths = {run_id: Path(data_path) / f"{run_id}.npz" for run_id in run_ids}
    datas: dict[int, pl.DataFrame] = {}
    for run_id, path in run_id_paths.items():
        if not path.exists():
            continue
        with np.load(path) as data:
            data_dict = {k: np.asarray(data[k]) for k in data.keys()}
        del data
        df = pl.DataFrame(data_dict)
        del data_dict
        df = df.with_columns(
            pl.lit(run_id).alias("id"),
            pl.lit(np.arange(len(df))).alias("frame"),
        )
        if "rewards" in df.columns and (
            metrics is None
            or not metrics
            or any(
                m in ["ewm_reward", "mean_ewm_reward", "mean_reward"] for m in metrics
            )
        ):
            df = calculate_ewm_reward(df)
            df = calculate_mean_reward(df)

        # Calculate object traces if requested
        if "object_collected_id" in df.columns and (
            metrics is None
            or not metrics
            or any(m.startswith("object_trace_") for m in metrics)
        ):
            df = calculate_object_traces(df)

        # Calculate biome occupancy if requested
        if "biome_id" in df.columns and (
            metrics is None
            or not metrics
            or any(m.startswith("biome_") for m in metrics)
        ):
            df = calculate_biome_occupancy(df)
        if start is not None or end is not None:
            start_idx = start if start is not None else 0
            length = (end - start_idx) if end is not None else None
            df = df.slice(start_idx, length)
        if sample is None:
            continue
        n = len(df)
        df = df.lazy()
        sample_types = [sample_type] if isinstance(sample_type, str) else sample_type
        datas_run_id = []
        for st in sample_types:
            if st == "every":
                df = df.gather_every(max(1, n // sample))
                df = df.with_columns(pl.lit(st).alias("sample_type"))
                datas_run_id.append(df)
            elif st == "random":
                df = df.collect().sample(n=sample, seed=0).sort("frame").lazy()
                df = df.with_columns(pl.lit(st).alias("sample_type"))
                datas_run_id.append(df)
            elif isinstance(st, tuple):
                if st[0] + st[1] > n:
                    continue
                df = df.slice(st[0], st[1] - st[0])
                if len(st) > 2:
                    step = n // st[2]
                    if step > 1:
                        df = df.gather_every(max(1, step))
                        df = df.with_columns(
                            pl.lit(f"{st[0]}:{st[1]}:{st[2]}").alias("sample_type")
                        )
                    else:
                        df = df.with_columns(
                            pl.lit(f"{st[0]}:{st[1]}").alias("sample_type")
                        )
                else:
                    df = df.with_columns(
                        pl.lit(f"{st[0]}:{st[1]}").alias("sample_type")
                    )
                datas_run_id.append(df)

        df = pl.concat(datas_run_id, how="diagonal").collect()
        datas[run_id] = df

    if len(datas) == 0:
        return pl.DataFrame().lazy()
    df = pl.concat(datas.values(), how="diagonal")
    del datas
    gc.collect()
    return df.lazy()


def load_all_results_from_data(
    data_path: str | Path,
    db_path: str | Path,
    metrics: Iterable[str] | None = None,
    ids: Iterable[int] | None = None,
    sample: int | None = 500,
    sample_type: str
    | tuple[int, int]
    | tuple[int, int, int]
    | list[str | tuple[int, int] | tuple[int, int, int]] = "every",
    start: int | None = None,
    end: int | None = None,
):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = sqlu.get_tables(cur)
    if metrics is None:
        metrics = tables - {"_metadata_"}

    df = read_metrics_from_data(
        data_path, metrics, ids, sample, sample_type, start, end
    )

    if "_metadata_" not in tables:
        return df

    meta = cx.read_sql(
        f"sqlite://{db_path}",
        "SELECT * FROM _metadata_",
        return_type="polars",
        partition_on="id",
        partition_num=1,
    )
    meta = meta.lazy()
    df = df.join(meta, how="left", on=["id"]).collect()
    del meta
    return df


class Result(Generic[Exp]):
    def __init__(
        self, exp_path: str | Path, exp: Exp, metrics: Sequence[str] | None = None
    ):
        self.exp_path = str(exp_path)
        self.exp = exp
        self.metrics = metrics

    def load(
        self,
        sample: int = 500,
        sample_type: str
        | tuple[int, int]
        | tuple[int, int, int]
        | list[str | tuple[int, int] | tuple[int, int, int]] = "every",
        start: int | None = None,
        end: int | None = None,
    ):
        db_path = self.exp.buildSaveContext(0).resolve("results.db")
        data_path = self.exp.buildSaveContext(0).resolve("data")

        if not Path(db_path).exists():
            if not Path(data_path).exists():
                return None
            df = read_metrics_from_data(
                data_path, self.metrics, None, sample, sample_type, start, end
            )
            df = df.with_columns(
                pl.col("id").alias("seed"),
            )
            return df.collect()

        run_ids = set()
        for param_id in range(self.exp.numPermutations()):
            params = getParamsAsDict(self.exp, param_id)
            run_ids.update(get_run_ids(db_path, params, data_path))
        run_ids = sorted(run_ids)
        df = load_all_results_from_data(
            data_path,
            db_path,
            self.metrics,
            run_ids,
            sample,
            sample_type,
            start,
            end,
        )
        return df

    def load_by_params(
        self, params: dict, start: int | None = None, end: int | None = None
    ):
        db_path = self.exp.buildSaveContext(0).resolve("results.db")
        data_path = self.exp.buildSaveContext(0).resolve("data")

        if not Path(db_path).exists():
            return None

        run_ids = get_run_ids(db_path, params)
        df = load_all_results_from_data(
            data_path,
            db_path,
            self.metrics,
            run_ids,
            start=start,
            end=end,
        )
        return df

    @property
    def filename(self):
        return self.exp_path.split("/")[-1].removesuffix(".json")


class ResultCollection(Generic[Exp]):
    def __init__(
        self,
        path: str | Path | None = None,
        metrics: Sequence[str] | None = None,
        Model: type[Exp] = ExperimentDescription,
    ):
        self.metrics = metrics
        self.Model = Model

        if path is None:
            main_file = importlib.import_module("__main__").__file__
            assert main_file is not None
            path = Path(main_file).parent

        self.path = Path(path)

        project = Path.cwd()
        paths = glob.glob(
            "**/*.json",
            root_dir=str(self.path),
            recursive=True,
        )
        paths = [str((self.path / p).absolute().relative_to(project)) for p in paths]
        self.paths = paths

    def _result(self, path: str):
        exp = loadExperiment(path, self.Model)
        return Result[Exp](path, exp, self.metrics)

    def get_hyperparameter_columns(self):
        hypers = set[str]()

        for path in self.paths:
            exp = loadExperiment(path, self.Model)
            hypers |= set(getHeader(exp))

        return sorted(hypers)

    def groupby_directory(self, level: int):
        uniques = set(p.split("/")[level] for p in self.paths)

        for group in uniques:
            group_paths = [p for p in self.paths if p.split("/")[level] == group]
            results = map(self._result, group_paths)
            yield group, list(results)

    def __iter__(self):
        return map(self._result, self.paths)


def detect_missing_indices(exp: ExperimentDescription, runs: int, base: str = "./"):
    context = exp.buildSaveContext(0, base=base)
    header = getHeader(exp)
    path = context.resolve("results.db")
    data_path = context.resolve("data")

    if not context.exists("results.db"):
        yield from listIndices(exp, runs)
        return

    n_params = exp.numPermutations()
    for param_id in range(n_params):
        run_ids = set(
            get_run_ids(path, getParamsAsDict(exp, param_id, header=header), data_path)
        )

        for seed in range(runs):
            run_id = seed * n_params + param_id
            if run_id not in run_ids:
                yield run_id


def gather_missing_indices(
    experiment_paths: Iterable[str],
    runs: int,
    loader: Callable[[str], ExperimentDescription] = loadExperiment,
    base: str = "./",
):
    path_to_indices: dict[str, list[int]] = {}

    for path in experiment_paths:
        exp = loader(path)
        indices = detect_missing_indices(exp, runs, base=base)
        indices = sorted(indices)
        path_to_indices[path] = indices

        size = exp.numPermutations() * runs
        print(path, f"{len(indices)} / {size}")

    return path_to_indices
